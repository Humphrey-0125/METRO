import os
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Callable, Optional

import numpy as np
import glob

def count_tokens(messages, tokenizer):
    return sum(len(tokenizer.encode(m["content"])) for m in messages)

def concat_states_text(states: Dict[str, Any]) -> str:
    parts = []
    for k, v in states.items():
        if k.endswith("_state") and isinstance(v, str) and v.strip():
            parts.append(f"{k}: {v.strip()}")
    return " | ".join(parts).strip()


_CLUSTER_CENTROIDS_CACHE: Dict[str, Dict[int, np.ndarray]] = {}
_CLUSTER_TO_CHAINS_CACHE: Dict[Tuple[str, str], Dict[int, List[List[str]]]] = {}
_CLUSTER_BEST_CHAIN_CACHE: Dict[str, List[List[str]]] = {}


def _normalize_strategy_chain(chain: Any) -> List[str]:
    """
    与 MCT.py 保持一致的策略链序列化逻辑：
      - 单个字符串：strip 后直接使用；
      - 列表/元组：表示并列策略，转成 "[a, b]" 这种格式，便于后续识别；
      - 其他类型：转成字符串兜底。
    """
    if not isinstance(chain, list):
        return []
    normalized: List[str] = []
    for step in chain:
        if isinstance(step, str):
            token = step.strip()
            if token:
                normalized.append(token)
            continue
        if isinstance(step, (list, tuple)):
            tokens = [str(tok).strip() for tok in step if isinstance(tok, str) and tok.strip()]
            if tokens:
                combo = "[" + ", ".join(tokens) + "]"
                normalized.append(combo)
            continue
        token = str(step).strip()
        if token:
            normalized.append(token)
    return normalized


def _standardize_tree_step(step: Any) -> str:
    """
    读取树节点时，将旧格式（使用 '->' 表示并列策略）的名称
    转换成统一的 '[a, b]' 格式，方便后续识别。
    """
    if isinstance(step, str):
        stripped = step.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            return stripped
        if "->" in stripped:
            parts = [p.strip() for p in stripped.split("->") if p.strip()]
            if len(parts) > 1:
                return "[" + ", ".join(parts) + "]"
            elif parts:
                return parts[0]
        return stripped
    return str(step)


def _parse_strategy_step(step: str) -> List[str]:
    """
    将字符串格式的策略步骤解析为策略列表。

    例如：
    - "[acknowledgement, credibility-appeal]" -> ["acknowledgement", "credibility-appeal"]
    - "single-strategy" -> ["single-strategy"]
    """
    step = step.strip()
    if step.startswith("[") and step.endswith("]"):
        content = step[1:-1].strip()
        if content:
            return [s.strip() for s in content.split(",") if s.strip()]
        else:
            return []
    else:
        return [step] if step else []

def load_cluster_centroids(clusters_path: str) -> Dict[int, np.ndarray]:
    """
    从 clusters.json 中加载每个聚类簇的 centroid 向量：
    返回: { cluster_id: np.ndarray([...]) }

    参数:
        clusters_path: clusters.json 的路径，例如:
            "outputs/P4G/cluster/cluster_k200/clusters.json"
    """
    global _CLUSTER_CENTROIDS_CACHE

    if clusters_path in _CLUSTER_CENTROIDS_CACHE:
        return _CLUSTER_CENTROIDS_CACHE[clusters_path]

    p = Path(clusters_path)
    if not p.exists():
        raise FileNotFoundError(f"clusters.json not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    centroids: Dict[int, np.ndarray] = {}
    for clu in data.get("clusters", []):
        cid = int(clu["cluster"])
        vec = np.asarray(clu["centroid"], dtype=np.float32)
        centroids[cid] = vec

    if not centroids:
        raise ValueError(f"No centroids loaded from {clusters_path}")

    _CLUSTER_CENTROIDS_CACHE[clusters_path] = centroids
    return centroids

def _get_node_metric(node: Dict[str, Any], metric: str = "avg_value") -> float:
    """
    从节点中取评价指标:
    - 优先用 node[metric];
    - 如果 metric == "avg_value" 且没有该字段，则回退为 value_sum / count;
    - metric == "success_rate" 且没有，则回退为 success / count;
    - 其他情况则为 0.0。
    """
    if metric in node:
        try:
            return float(node[metric])
        except Exception:
            pass

    cnt = max(int(node.get("count", 0)), 1)
    if metric == "avg_value":
        vs = float(node.get("value_sum", 0.0))
        return vs / cnt
    if metric == "success_rate":
        succ = float(node.get("success", 0.0))
        return succ / cnt
    return 0.0


def _find_best_path_from_tree(
    node: Dict[str, Any],
    metric: str = "avg_value",
    path_prefix: List[List[str]] = None,
) -> Tuple[List[List[str]], float]:
    """
    在一棵 MCT 上做 DFS，找到指定 metric 最高的叶子节点路径。

    返回:
        (best_path, best_score)
        - best_path: 从 ROOT 下方到最佳叶子节点的策略序列（不含 "ROOT"），每个元素是策略列表
        - best_score: 该叶子节点的 metric 值
    """
    if path_prefix is None:
        path_prefix = []

    raw_name = node.get("name", "")
    name = _standardize_tree_step(raw_name)
    children = node.get("children", []) or []

    if name == "ROOT":
        current_path = []
    else:
        strategy_list = _parse_strategy_step(name)
        current_path = path_prefix + [strategy_list]

    if not children:
        score = _get_node_metric(node, metric=metric)
        return current_path, score

    best_path = None
    best_score = float('-inf')

    for ch in children:
        child_path, child_score = _find_best_path_from_tree(
            ch,
            metric=metric,
            path_prefix=current_path,
        )
        if child_score > best_score:
            best_score = child_score
            best_path = child_path

    if best_path is not None:
        return best_path, best_score

    score = _get_node_metric(node, metric=metric)
    return current_path, score


def load_best_chains_from_trees_dir(
    trees_dir: str,
    metric: str = "avg_value",
) -> List[List[List[str]]]:
    """
    扫描某个 trees_value 目录下的所有 cluster_*.json，
    为每个 cluster 选出一条"价值最高"的策略链。

    返回:
        List[best_chain]，其中 best_chain 是 List[List[str]]，
        列表索引即 cluster_id，每条链的每个步骤都是策略列表，例如:
        result[0] = [["acknowledgement"], ["credibility-appeal"], ["ask-donation-amount"], ...]
    """
    global _CLUSTER_BEST_CHAIN_CACHE
    if trees_dir in _CLUSTER_BEST_CHAIN_CACHE:
        return _CLUSTER_BEST_CHAIN_CACHE[trees_dir]

    pattern = str(Path(trees_dir) / "cluster_*.json")
    cids = []
    for path in glob.glob(pattern):
        p = Path(path)
        stem = p.stem  # "cluster_0"
        try:
            cid = int(stem.split("_")[-1])
            cids.append(cid)
        except Exception:
            continue

    if not cids:
        return []

    max_cid = max(cids)
    result = [[] for _ in range(max_cid + 1)]

    for path in glob.glob(pattern):
        p = Path(path)
        stem = p.stem  # "cluster_0"
        try:
            cid = int(stem.split("_")[-1])
        except Exception:
            continue

        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)

        tree = data.get("tree")
        if not isinstance(tree, dict):
            continue

        best_path, best_score = _find_best_path_from_tree(tree, metric=metric)
        # print(f"[debug] cluster {cid}: best_{metric}={best_score}, path={best_path}")
        result[cid] = best_path

    _CLUSTER_BEST_CHAIN_CACHE[trees_dir] = result
    return result


def load_cluster_to_chains(
    clusters_path: str,
    states_path: str,
) -> Dict[int, List[List[str]]]:
    """
    构建: cluster_id -> 该簇下所有状态对应的 strategy_chain 列表。
    """
    global _CLUSTER_TO_CHAINS_CACHE
    cache_key = (clusters_path, states_path)

    if cache_key in _CLUSTER_TO_CHAINS_CACHE:
        return _CLUSTER_TO_CHAINS_CACHE[cache_key]

    if not os.path.exists(states_path):
        raise FileNotFoundError(f"states_embeddings file not found: {states_path}")
    with open(states_path, "r", encoding="utf-8") as f:
        states_data = json.load(f)

    id2chain: Dict[str, List[str]] = {}
    for obj in states_data:
        rid = obj.get("id")
        chain = obj.get("strategy_chain")
        if isinstance(rid, str) and chain:
            normalized = _normalize_strategy_chain(chain)
            if normalized:
                id2chain[rid] = normalized

    p = Path(clusters_path)
    if not p.exists():
        raise FileNotFoundError(f"clusters.json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        clu_data = json.load(f)

    cluster_to_chains: Dict[int, List[List[str]]] = {}
    for clu in clu_data.get("clusters", []):
        cid = int(clu["cluster"])
        members = clu.get("members", []) or []
        chains: List[List[str]] = []
        for mid in members:
            ch = id2chain.get(mid)
            if ch:
                chains.append(ch)
        cluster_to_chains[cid] = chains

    _CLUSTER_TO_CHAINS_CACHE[cache_key] = cluster_to_chains
    return cluster_to_chains

def _extract_aspect_dict(aspect_states: Any) -> Dict[str, str]:
    """
    把 analyze_aspects 的返回统一成 dict 形式:
    - 如果本身是 dict，则直接转成 {k: str(v)}
    - 如果是其他结构，则放到 "raw" 里兜底
    """
    if isinstance(aspect_states, dict):
        return {k: str(v) for k, v in aspect_states.items()}
    return {"raw": str(aspect_states)}


def _aspect_top_clusters_by_embedding(
    aspect_states: Any,
    emb_fn: Callable[[str], np.ndarray],
    clusters_path: str,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
    """
    利用 aspect 的 embedding 和 cluster centroid 做相似度匹配:
      1) 使用 concat_states_text 将 aspect 拼成文本;
      2) 用外部传入的 emb_fn 编码为向量;
      3) 与每个 cluster 的 centroid 做 cosine 相似度;
      4) 返回 top_k 个最相似簇 (cluster_id, similarity)。

    参数:
        aspect_states: analyze_aspects 返回的状态 (dict 或其他)
        emb_fn: 文本 -> 向量的函数，由外部提供
        clusters_path: clusters.json 文件路径
        top_k: 需要返回的簇个数 (默认 3)

    返回:
        List[(cluster_id, similarity)]
    """
    aspect_dict = _extract_aspect_dict(aspect_states)
    query_text = concat_states_text(aspect_dict)  # e.g. "value_state: ... | stage_state: ... | persona_state: ..."
    if not query_text:
        return []

    q_emb = emb_fn(query_text)
    if q_emb is None:
        return []

    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_emb) + 1e-8)

    centroids = load_cluster_centroids(clusters_path)

    sims: List[Tuple[int, float]] = []
    for cid, c_vec in centroids.items():
        c_norm = float(np.linalg.norm(c_vec) + 1e-8)
        sim = float(np.dot(q_emb, c_vec) / (q_norm * c_norm))
        sims.append((cid, sim))

    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def retrieve_strategy_chain_by_aspects(
    current_states: Any,
    emb_fn: Callable[[str], np.ndarray],
    clusters_path: str,
    trees_dir: str,
    top_k: int = 1,
) -> List[List[List[str]]]:
    """
    根据当前对话的 aspect，从历史中检索最相似的“最佳策略链”。

    流程:
      1) 使用 concat_states_text 拼接 aspect 文本;
      2) emb_fn 将文本编码为向量;
      3) 利用簇 centroid 找到 top-3 相似 cluster;
      4) 对每个 cluster 取出其 MCT 中“价值最高”的 strategy_chain，最多返回 top_k 条。

    参数:
        current_states: analyze_aspects 的结果
        emb_fn: 文本 -> 向量的函数，由外部模块实现并传入
        clusters_path: clusters.json 路径
        trees_dir: 保存 MCT 的目录，如: "outputs/P4G/cluster/cluster_k200/trees_value"
        top_k: 最多返回多少条策略链

    返回:
        List[ strategy_chain ]，其中 strategy_chain 是 List[List[str]]，
        每个策略步骤都是策略列表（如[["acknowledgement"], ["credibility-appeal", "emotion-appeal"]]）
    """
    top_clusters = _aspect_top_clusters_by_embedding(
        aspect_states=current_states,
        emb_fn=emb_fn,
        clusters_path=clusters_path,
        top_k=3,
    )
    if not top_clusters:
        return []

    cluster_best_chains = load_best_chains_from_trees_dir(
        trees_dir=trees_dir,
        metric="avg_value",
    )

    results: List[List[List[str]]] = []
    seen = set()

    for cid, sim in top_clusters:
        if cid >= len(cluster_best_chains):
            continue
        nested_chain = cluster_best_chains[cid]  # List[List[str]]
        if not nested_chain:
            continue

        tup = tuple(tuple(step) for step in nested_chain)
        if tup in seen:
            continue
        seen.add(tup)
        results.append(nested_chain)
        if len(results) >= top_k:
            break

    return results


def retrieve_strategy_chain_by_history(
    dialog_history: List[Dict[str, Any]],
    emb_fn: Callable[[str], np.ndarray],
    clusters_path: str,
    trees_dir: str,
    current_turn_id: int,
    top_k: int = 2,
    max_history_turns: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    根据当前对话的历史，从历史中检索最相似的"最佳策略链"。

    流程:
      1) 使用 concat_history_utterances_from_map 拼接历史对话文本;
      2) emb_fn 将文本编码为向量;
      3) 利用簇 centroid 找到最相似的一个 cluster;
      4) 根据该 cluster id 读取 topk 目录中对应文件的最佳策略链。

    参数:
        dialog_history: 当前的对话历史
        emb_fn: 文本 -> 向量的函数，由外部模块实现并传入
        clusters_path: clusters.json 路径
        trees_dir: 现在应为 topk 目录，如:
            "outputs/P4G/cluster/history/kmeans_k150/topk"
        current_turn_id: 当前轮次 ID，用于确定历史范围
        top_k: 最多返回多少条策略链，从每个簇的topk文件中选择
        max_history_turns: 可选，限制历史轮数

    返回:
        List[Dict[str, Any]]，每个字典包含：
        - "chain": List[List[str]]，策略链（每个策略步骤都是策略列表）
        - "cluster_id": int，簇ID
        - "similarity": float，相似度分数
    """
    from src.state.history_state_chain import build_turn_map, concat_history_utterances_from_map

    turn_map = build_turn_map(dialog_history)
    history_text = concat_history_utterances_from_map(turn_map, current_turn_id, max_history_turns)

    if not history_text:
        return []

    q_emb = emb_fn(history_text)
    if q_emb is None:
        return []

    q_emb = np.asarray(q_emb, dtype=np.float32)
    q_norm = float(np.linalg.norm(q_emb) + 1e-8)

    centroids = load_cluster_centroids(clusters_path)

    import time
    time1=time.time()

    sims: List[Tuple[int, float]] = []
    for cid, c_vec in centroids.items():
        c_norm = float(np.linalg.norm(c_vec) + 1e-8)
        sim = float(np.dot(q_emb, c_vec) / (q_norm * c_norm))
        sims.append((cid, sim))

    if not sims:
        return []

    sims.sort(key=lambda x: x[1], reverse=True)
    best_cid, best_sim = sims[0]
    time2=time.time()
    print(f"Strategy chain retrieval time: {time2-time1:.2f}s")

    topk_dir = Path(trees_dir)
    topk_file = topk_dir / f"cluster_{best_cid}_top3.json"
    if not topk_file.exists():
        return []

    try:
        with topk_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return []

    top_chains = data.get("top_chains") or []
    if not top_chains:
        return []

    selected_chains = top_chains[:top_k]
    result = []

    for i, entry in enumerate(selected_chains):
        raw_chain = entry.get("chain") or []
        if not isinstance(raw_chain, list) or not raw_chain:
            continue

        parsed_chain: List[List[str]] = []
        for step in raw_chain:
            if not isinstance(step, str):
                continue
            parsed = _parse_strategy_step(step)
            if parsed:
                parsed_chain.append(parsed)

        if not parsed_chain:
            continue

        result.append({
            "chain": parsed_chain,
            "cluster_id": best_cid,
            "cluster": best_cid,
            "cluster_idx": best_cid,
            "similarity": best_sim,
            "rank": i,
        })

    return result
