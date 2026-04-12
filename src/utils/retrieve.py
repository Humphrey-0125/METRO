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
    if isinstance(aspect_states, dict):
        return {k: str(v) for k, v in aspect_states.items()}
    return {"raw": str(aspect_states)}


def _aspect_top_clusters_by_embedding(
    aspect_states: Any,
    emb_fn: Callable[[str], np.ndarray],
    clusters_path: str,
    top_k: int = 3,
) -> List[Tuple[int, float]]:
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
