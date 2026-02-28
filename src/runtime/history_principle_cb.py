#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CB Principle-guided bargaining simulation (history-based retrieval).

核心逻辑（CB版）：
- 用对话历史 embedding 检索最相似簇 / MCT 策略路径（retrieve_strategy_chain_by_history）；
- 在该簇内，基于最近一条 seller utterance（get_last_utterance("seller")）检索 top-5 principles；
  * 检索 key = principle_text 的 WHEN 子句（逗号前）；
- 将策略链提示 + principle 高层指导传给 buyer 生成回复；
- 用 critic_model_cb 判断是否成交、成交价，决定是否结束；
- 增量保存与断点续传：对齐你 P4G 版本的 IncrementalSaver 思路（但 key 用 dialogue_id）。
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from filelock import FileLock

# =============== CB judge (对齐 raw_model_cb.py) ===============
from src.evaluate.critic_model_cb import call_critic_model

# =============== retrieval / embedding ===============
from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL
from src.utils.retrieve import retrieve_strategy_chain_by_history

from src.utils.llm_dialog import (
    chat_completion_buyer,
    chat_completion_seller,
    history_to_plain_text,
)
# =============== guidance helpers (你已有的) ===============
from src.utils.guidance_cb import generate_high_level_guidance, generate_strategy_chain_summary

# =============== 如果你想沿用 format_strategy_hint，可继续用 ===============
from src.utils.llm_dialog import format_strategy_hint

# =============== 路径配置（CB版） ===============
DEFAULT_CLUSTER_CONFIG = "kmeans_k80"

CLUSTERS_PATH = "outputs/CB/cluster/history/{}/clusters.json"
TREES_DIR     = "outputs/CB/cluster/history/{}/topk"
PRINCIPLES_DIR = "outputs/CB/cluster/history/{}/principles_by_cluster"

# CLUSTERS_PATH = "outputs/P4G/cluster/history/{}/clusters.json"
# TREES_DIR     = "outputs/P4G/cluster/history/{}/topk"
# PRINCIPLES_DIR = "outputs/P4G/cluster/history/{}/principles_by_cluster"

# CLUSTERS_PATH = "outputs/ALL/cluster/history/{}/clusters.json"
# TREES_DIR     = "outputs/ALL/cluster/history/{}/topk"
# PRINCIPLES_DIR = "outputs/ALL/cluster/history/{}/principles_by_cluster"

MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo-0125")

# =============== 数据路径（对齐 raw_model_cb.py） ===============
DEFAULT_DEV_PATH = os.getenv("CB_DEV_PATH", "data/CB/dev.json")
DEFAULT_PERSONA_PATH = os.getenv("CB_PERSONA_PATH", "outputs/P4G/personas/personas_eval.jsonl")

# =============== 全局 Embedder 与缓存 ===============
_EMBEDDER: Optional[Embedder] = None
_CLUSTER_PRINCIPLES_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CLUSTER_PRINCIPLES_EMB_CACHE: Dict[str, List[List[float]]] = {}

# =============== 生成对话内容 ===============
def generate_buyer_utterance_cb_ours(
    dialog_history: List[Dict[str, Any]],
    meta: Dict[str, Any],
    strategy_chain_hint: Optional[str],
    guidance_text: Optional[str],
    prompt_type: str,
    model: str,
    temperature: float = 0.0,
    max_tokens: Optional[int] = 96,
) -> str:
    """
    Generate CB buyer utterance with ablation-aware prompt construction.

    - depth  = strategy_chain_hint (long-term planning)
    - breadth= guidance_text (high-level guidance)
    If either is empty/None, we treat it as ablated and REMOVE it structurally
    via flags passed into the prompt builder.
    """
    from prompts.cb.buyer.ours import build_buyer_generation_prompt_cb_ours

    dialogue_text = history_to_plain_text(dialog_history)

    def _is_effectively_empty(x: Optional[str]) -> bool:
        return x is None or (isinstance(x, str) and x.strip() == "")

    use_depth = not _is_effectively_empty(strategy_chain_hint)
    use_breadth = not _is_effectively_empty(guidance_text)

    # For structural ablation: pass None when disabled (not ""), and pass explicit flags.
    strategy_chain_hint_to_use = strategy_chain_hint if use_depth else None
    guidance_text_to_use = guidance_text if use_breadth else None

    sys_p, usr_p = build_buyer_generation_prompt_cb_ours(
        dialogue_text=dialogue_text,
        item_name=meta["item_name"],
        buyer_target_price=meta["buyer_price"],
        item_description=meta["buyer_item_description"],
        strategy_chain_hint=strategy_chain_hint_to_use,
        guidance_text=guidance_text_to_use,
        prompt_type=prompt_type,
        use_depth=use_depth,         # ✅ NEW: ablation control
        use_breadth=use_breadth,     # ✅ NEW: ablation control
    )

    messages = [
        {"role": "system", "content": sys_p},
        {"role": "user", "content": usr_p},
    ]
    print("[messages]: ", messages)
    out = chat_completion_buyer(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    return (out or "").strip()


def generate_seller_utterance_cb_persona(
    dialog_history: List[Dict[str, Any]],
    meta: Dict[str, Any],
    persona_description: str,
    prompt_type: str,
    model: str,
    temperature: float = 0.5,
    max_tokens: Optional[int] = 96,
) -> str:
    # 先沿用 raw_model_cb.py 的 persona prompt
    from prompts.cb.seller.persona import build_seller_generation_prompt_persona

    dialogue_text = history_to_plain_text(dialog_history)

    sys_p, usr_p = build_seller_generation_prompt_persona(
        dialogue_text=dialogue_text,
        persona_description=persona_description,
        item_name=meta["item_name"],
        seller_target_price=meta["seller_price"],
        item_description=meta["seller_item_description"],
    )
    messages = [{"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}]
    out = chat_completion_seller(messages, model=model, temperature=temperature, max_tokens=max_tokens)
    return (out or "").strip()


def get_embedder() -> Embedder:
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(
            model_name=EMBED_MODEL_NAME,
            api_key=EMBED_API_KEY,
            api_url=EMBED_API_URL,
        )
    return _EMBEDDER


def emb_fn(text: str) -> List[float]:
    return get_embedder().encode(text or "")


# ===================== principles 工具 =====================

def extract_when_clause(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    idx = text.find(",")
    return text if idx == -1 else text[:idx].strip()


def load_principles_for_cluster(cluster_id: int, cluster_config: str = DEFAULT_CLUSTER_CONFIG) -> List[Dict[str, Any]]:
    """
    读取某个簇的所有 principles（带缓存）
    文件路径：PRINCIPLES_DIR.format(cluster_config)/cluster_{cluster_id}.json
    """
    cache_key = f"{cluster_config}_{cluster_id}"
    if cache_key in _CLUSTER_PRINCIPLES_CACHE:
        return _CLUSTER_PRINCIPLES_CACHE[cache_key]

    principles_dir = PRINCIPLES_DIR.format(cluster_config)
    path = os.path.join(principles_dir, f"cluster_{cluster_id}.json")
    if not os.path.exists(path):
        print(f"[warn] principles file not found for cluster {cluster_id}: {path}")
        _CLUSTER_PRINCIPLES_CACHE[cache_key] = []
        _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = []
        return []

    with open(path, "r", encoding="utf-8") as f:
        items = json.load(f)

    cleaned: List[Dict[str, Any]] = []
    for it in items:
        cleaned.append({
            "id": it.get("id"),
            "principle_type": it.get("principle_type"),
            "principle_text": it.get("principle_text"),
        })

    _CLUSTER_PRINCIPLES_CACHE[cache_key] = cleaned
    _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = []  # 延迟 embedding
    return cleaned


def dot_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def get_top_principles_for_cluster(
    cluster_id: int,
    query_text: str,
    top_k: int = 5,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
) -> List[Dict[str, Any]]:
    if not query_text:
        return []

    principles = load_principles_for_cluster(cluster_id, cluster_config)
    if not principles:
        return []

    cache_key = f"{cluster_config}_{cluster_id}"

    if not _CLUSTER_PRINCIPLES_EMB_CACHE.get(cache_key, []):
        emb_list: List[List[float]] = []
        for p in principles:
            when_clause = extract_when_clause(p.get("principle_text") or "")
            emb_list.append(emb_fn(when_clause))
        _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = emb_list

    query_emb = emb_fn(query_text)
    cand_embs = _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key]

    sims = []
    for idx, emb in enumerate(cand_embs):
        sims.append((dot_similarity(query_emb, emb), idx))
    sims.sort(key=lambda x: x[0], reverse=True)

    results: List[Dict[str, Any]] = []
    for sim, idx in sims[:top_k]:
        item = principles[idx].copy()
        item["similarity"] = sim
        results.append(item)
    return results


def get_last_utterance(dialog_history: List[Dict[str, Any]], speaker: str) -> str:
    """
    取最近一条指定 speaker 的 text（CB：speaker="seller"）
    """
    for turn in reversed(dialog_history):
        if (turn.get("speaker") or "").lower() == speaker.lower():
            txt = turn.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return ""


# ===================== CB 数据加载（对齐 raw_model_cb.py） =====================

def load_json_auto(path: str) -> Any:
    """
    支持 JSON / JSONL
    """
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []

    if "\n" in content and not content.lstrip().startswith("["):
        data = []
        for line_no, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data

    return json.loads(content)


def load_cb_dev_dataset(dev_path: str = DEFAULT_DEV_PATH) -> List[Dict[str, Any]]:
    data = load_json_auto(dev_path)
    if not isinstance(data, list):
        raise ValueError("CB dev.json top-level must be a list")
    return data


def load_cb_personas(persona_path: str = DEFAULT_PERSONA_PATH) -> List[str]:
    personas = load_json_auto(persona_path)
    if not isinstance(personas, list):
        raise ValueError("CB persona file top-level must be a list (jsonl -> list)")
    persona_list: List[str] = []
    for p in personas:
        if isinstance(p, dict):
            persona_list.append(p.get("description") or p.get("persona") or "")
        else:
            persona_list.append(str(p))
    return persona_list


# ===================== 增量保存（CB：按 dialogue_id 去重） =====================

class IncrementalSaver:
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.lock_file = output_file + ".lock"
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def _read_all(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.output_file):
            return []
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def save_result(self, result: Dict[str, Any], key: str = "dialogue_id"):
        with FileLock(self.lock_file, timeout=30):
            all_data = self._read_all()
            rid = result.get(key)

            replaced = False
            if rid is not None:
                for i, r in enumerate(all_data):
                    if r.get(key) == rid:
                        all_data[i] = result
                        replaced = True
                        break
            if not replaced:
                all_data.append(result)

            tmp = self.output_file + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(all_data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, self.output_file)

    def is_completed(self, rid: str, key: str = "dialogue_id") -> bool:
        with FileLock(self.lock_file, timeout=10):
            all_data = self._read_all()
            return any(r.get(key) == rid for r in all_data)

    def get_completed_count(self) -> int:
        with FileLock(self.lock_file, timeout=10):
            return len(self._read_all())


# ===================== CB round counting =====================

def count_rounds(dialog_history: List[Dict[str, Any]]) -> int:
    tids = []
    for t in dialog_history:
        try:
            tids.append(int(t.get("turn_id", -1)))
        except Exception:
            continue
    return (max(tids) + 1) if tids else 0


# ===================== CB judge（对齐 raw_model_cb.py） =====================

def call_cb_judge(dialog_history: List[Dict[str, Any]], max_retries: int = 3) -> Dict[str, Any]:
    deal, should_end, price, details = call_critic_model(
        dialog_history=dialog_history,
        prompt_type="DPDP",
        num_samples=3,
        max_retries=max_retries,
        return_details=True,
    )
    return {"deal": bool(deal), "price": price if deal else None, "details": details, "should_end": bool(should_end)}


# ===================== 初始 turn0（对齐 raw_model_cb.py） =====================

def build_initial_turn0(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    item_name = meta["item_name"]
    seller_price = meta["seller_price"]
    return [
        {"turn_id": 0, "speaker": "buyer", "text": f"Hi, how much is the {item_name}?"},
        {"turn_id": 0, "speaker": "seller", "text": f"Hi, this is a good {item_name} and its price is {seller_price}."},
    ]

def get_first_strategy_group_from_retrieved(retrieved_chains: List[Any]) -> str:
    """
    从 top-1 retrieved_chains 中取 chain 的第一层策略组。
    chain 格式: List[List[str]]，第一层可能多个策略。
    返回: "ask_question" 或 "ask_question | build_rapport"
    失败返回空串。
    """
    if not retrieved_chains:
        return ""
    c0 = retrieved_chains[0]
    if not isinstance(c0, dict):
        return ""
    chain = c0.get("chain")
    if not isinstance(chain, list) or not chain:
        return ""
    first_group = chain[0]
    if not isinstance(first_group, list) or not first_group:
        return ""
    parts = [str(x).strip() for x in first_group if str(x).strip()]
    return " | ".join(parts) if parts else ""

# ===================== 主模拟流程（CB版） =====================

BUYER_PROMPT_TYPE = "Ours_1"      # 先占位
SELLER_PROMPT_TYPE = "personas"  # 先占位

def run_simulation_one_dialog(
    dialogue_id: str,
    meta: Dict[str, Any],
    seller_persona: str,
    max_turns: int = 9,
    buyer_prompt_type: Optional[str] = None,
    seller_prompt_type: Optional[str] = None,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
    verbose: bool = False,
    max_retries: int = 3,
    ablation_mode: str = "none",  # "none" | "w/o_depth" | "w/o_breadth" | "w/o_both" | "w/o_expend"
) -> Dict[str, Any]:
    """
    Pure ablation:
    - depth   = strategy chain hint (strategy_hint_text / strategy_chain_summary)
    - breadth = high-level guidance generated from principles (expand)

    Modes:
    - none:        depth + breadth (full)
    - w/o_depth:   breadth only (skip chain summary; do not feed chain hint)
    - w/o_breadth: depth only (skip guidance generation; do not feed guidance)
    - w/o_both:    neither
    - w/o_expend:  still 2-view, but NO principle->guidance expansion:
                  Immediate view = first strategy group from retrieved chain
    """
    if buyer_prompt_type is None:
        buyer_prompt_type = BUYER_PROMPT_TYPE
    if seller_prompt_type is None:
        seller_prompt_type = SELLER_PROMPT_TYPE

    # 价格 sanity check（建议早点暴露数据异常）
    bp = meta.get("buyer_price", None)
    sp = meta.get("seller_price", None)
    if bp is not None and sp is not None:
        try:
            assert float(sp) > float(bp), f"{dialogue_id} invalid: seller_price={sp} <= buyer_price={bp}"
        except Exception:
            pass

    dialog_history = build_initial_turn0(meta)
    current_turn_id = 1

    # --------- ablation flags (pure) ----------
    mode = (ablation_mode or "none").strip()
    valid_modes = {"none", "w/o_depth", "w/o_breadth", "w/o_both", "w/o_expend"}
    if mode not in valid_modes:
        mode = "none"

    disable_depth = mode in {"w/o_depth", "w/o_both"}
    disable_breadth = mode in {"w/o_breadth", "w/o_both"}
    use_first_group_as_immediate = (mode == "w/o_expend")

    while True:
        # ---- 1) 检索策略链（top-1）----
        clusters_path = CLUSTERS_PATH.format(cluster_config)
        trees_dir = TREES_DIR.format(cluster_config)

        try:
            retrieved_chains = retrieve_strategy_chain_by_history(
                dialog_history=dialog_history,
                emb_fn=emb_fn,
                clusters_path=clusters_path,
                trees_dir=trees_dir,
                current_turn_id=current_turn_id,
                top_k=1,  # 返回前3个候选策略链
            )
        except Exception as e:
            print(f"[warn] retrieve_strategy_chain_by_history failed: {e}")
            retrieved_chains = []

        # depth 原始文本形态（可读链）
        strategy_hint_text = format_strategy_hint(retrieved_chains,depth=3) or ""
        # print(f"strategy_hint_text: {strategy_hint_text}")
        # print(f"retrieved_chains: {retrieved_chains}")
        # ---- 2) principles：用最近 seller utterance 检索 ----
        cluster_id = None
        if retrieved_chains:
            c0 = retrieved_chains[0]
            if isinstance(c0, dict):
                cluster_id = c0.get("cluster_id") or c0.get("cluster") or c0.get("cluster_idx")

        query_text = get_last_utterance(dialog_history, speaker="seller")

        top_principles: List[Dict[str, Any]] = []
        if isinstance(cluster_id, int):
            top_principles = get_top_principles_for_cluster(
                cluster_id=cluster_id,
                query_text=query_text,
                top_k=5,
                cluster_config=cluster_config,
            )

        # ---- 3) depth/breadth inputs（纯消融控制）----
        # 注意：query_text 必须是 “卖家上一句”
        query_text = get_last_utterance(dialog_history, speaker="seller")

        # recent dialogue（用于 Ours_1 的 summary 或 guidance）
        from src.utils.llm_dialog import history_to_plain_text
        recent_dialogue = history_to_plain_text(dialog_history)
        # if len(dialog_history) >= 2:
        #     recent_dialogue = history_to_plain_text(dialog_history[-2:])
        # elif len(dialog_history) >= 1:
        #     recent_dialogue = history_to_plain_text(dialog_history[-1:])
        # else:
        #     recent_dialogue = ""

        # ---------- breadth (immediate / guidance) ----------
        breadth_guidance_to_feed = ""

        if use_first_group_as_immediate:
            # ✅ w/o_expend：不调用 generate_high_level_guidance
            first_group = get_first_strategy_group_from_retrieved(retrieved_chains)
            if first_group:
                breadth_guidance_to_feed = f"Immediate suggestion (use this now): {first_group}"
            else:
                breadth_guidance_to_feed = ""
        else:
            # 原逻辑：breadth 来自 principles -> guidance（除非 w/o_breadth/w/o_both）
            if not disable_breadth:
                if buyer_prompt_type == "Ours_1":
                    breadth_guidance_to_feed = generate_high_level_guidance(
                        principles=top_principles,
                        last_user_utt=query_text,  # ✅ seller last utt
                        recent_dialogue=recent_dialogue,
                        model=MODEL,
                        temperature=0.5,
                        max_tokens=100
                    )
                else:
                    breadth_guidance_to_feed = generate_high_level_guidance(
                        principles=top_principles,
                        last_user_utt=query_text,  # ✅ seller last utt
                        model=MODEL,
                        temperature=0.5,
                        max_tokens=100
                    )
            else:
                breadth_guidance_to_feed = ""

        # ---------- depth (strategy chain hint) ----------
        depth_hint_to_feed = ""
        used_strategy_hint = ""

        if disable_depth:
            depth_hint_to_feed = ""
            used_strategy_hint = ""
        else:
            if buyer_prompt_type == "Ours_1":
                strategy_chain_summary = generate_strategy_chain_summary(
                    strategy_chain_hint=strategy_hint_text,
                    recent_dialogue=recent_dialogue,
                    model=MODEL,
                    temperature=0.5,
                    max_tokens=100
                )
                depth_hint_to_feed = strategy_chain_summary
                used_strategy_hint = strategy_chain_summary
            else:
                depth_hint_to_feed = strategy_hint_text
                used_strategy_hint = strategy_hint_text

        # ---- 3.5) buyer utterance（ablation 生效于两个输入）----
        buyer_text = generate_buyer_utterance_cb_ours(
            dialog_history=dialog_history,
            meta=meta,
            strategy_chain_hint=depth_hint_to_feed,          # depth
            guidance_text=breadth_guidance_to_feed,          # breadth / immediate
            prompt_type=buyer_prompt_type,
            model=MODEL,
            temperature=0.0,
            max_tokens=96,
        )

        buyer_turn = {
            "turn_id": current_turn_id,
            "speaker": "buyer",
            "text": buyer_text,
            "strategy_hint": retrieved_chains,
            "strategy_hint_text": used_strategy_hint,              # summary/原始/空
            "principle_hint": top_principles,
            "high_level_guidance": breadth_guidance_to_feed,       # guidance/first-group/空
            "ablation_mode": mode,
        }
        dialog_history.append(buyer_turn)

        # ---- 4) seller reply（persona seller）----
        seller_text = generate_seller_utterance_cb_persona(
            dialog_history=dialog_history,
            meta=meta,
            persona_description=seller_persona,
            prompt_type=seller_prompt_type,
            model=MODEL,
            temperature=0.5,
            max_tokens=96,
        )
        dialog_history.append({"turn_id": current_turn_id, "speaker": "seller", "text": seller_text})

        if verbose:
            print(f"\n[dialogue_id={dialogue_id}] turn={current_turn_id} mode={mode}")
            print("[buyer]:", buyer_text)
            print("[seller]:", seller_text)
            print("------------------------------------------------")

        # ---- 5) judge ----
        outcome = call_cb_judge(dialog_history, max_retries=max_retries)

        if outcome["deal"]:
            return {
                "dialogue_id": dialogue_id,
                "num_turns": count_rounds(dialog_history),
                "deal": True,
                "price": outcome["price"],
                "dialog": dialog_history,
                "details": outcome["details"],
                "meta": meta,
                "ablation_mode": mode,
            }

        # ---- 6) stop? ----
        if outcome.get("should_end", False) or current_turn_id >= max_turns - 1:
            break

        current_turn_id += 1

    return {
        "dialogue_id": dialogue_id,
        "num_turns": count_rounds(dialog_history),
        "deal": False,
        "price": None,
        "dialog": dialog_history,
        "details": [],
        "meta": meta,
        "ablation_mode": mode,
    }


# ===================== Batch evaluation（CB版，对齐 raw_model_cb.py 的索引方式） =====================

def run_batch_evaluation(
    dev_path: str = DEFAULT_DEV_PATH,
    persona_path: str = DEFAULT_PERSONA_PATH,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
    max_turns: int = 9,
    max_samples: Optional[int] = None,
    start_index: int = 0,
    end_index: Optional[int] = None,
    output_file: Optional[str] = None,
    enable_incremental_save: bool = True,
    verbose: bool = False,
    max_retries: int = 3,
    ablation_mode: str = "none",  # ✅ 新增：none | w/o_depth | w/o_breadth | w/o_both
) -> List[Dict[str, Any]]:

    dev_data = load_cb_dev_dataset(dev_path)
    persona_list = load_cb_personas(persona_path)

    n = min(len(dev_data), len(persona_list))
    if max_samples is not None:
        n = min(n, int(max_samples))
    if n <= 0:
        return []

    if end_index is None:
        end_index = n - 1
    start_index = max(0, int(start_index))
    end_index = min(n - 1, int(end_index))

    if output_file is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_mode = ablation_mode.replace("/", "_").replace(" ", "")
        output_file = f"outputs/CB/evaluate/history/{cluster_config}/results_{safe_mode}_{ts}.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    saver = IncrementalSaver(output_file) if enable_incremental_save else None

    skipped = 0
    failed = 0

    indices = []
    for i in range(start_index, end_index + 1):
        sample = dev_data[i]
        dialogue_id = sample.get("dialogue_id") or sample.get("id") or f"cb_{i:06d}"
        if saver and saver.is_completed(str(dialogue_id)):
            skipped += 1
            continue
        indices.append(i)

    total = (end_index - start_index + 1)

    if len(indices) == 0:
        if saver and os.path.exists(saver.output_file):
            with FileLock(saver.lock_file, timeout=10):
                with open(saver.output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
        return []

    with tqdm(total=total, desc="CB history+principle", unit="dlg", initial=skipped) as pbar:
        for i in indices:
            sample = dev_data[i]
            seller_persona = persona_list[i]
            dialogue_id = sample.get("dialogue_id") or sample.get("id") or f"cb_{i:06d}"

            meta = {
                "item_name": sample.get("item_name", ""),
                "buyer_item_description": sample.get("buyer_item_description", ""),
                "buyer_price": sample.get("buyer_price", None),
                "seller_item_description": sample.get("seller_item_description", ""),
                "seller_price": sample.get("seller_price", None),
            }

            try:
                result = run_simulation_one_dialog(
                    dialogue_id=str(dialogue_id),
                    meta=meta,
                    seller_persona=seller_persona,
                    max_turns=max_turns,
                    buyer_prompt_type=BUYER_PROMPT_TYPE,
                    seller_prompt_type=SELLER_PROMPT_TYPE,
                    cluster_config=cluster_config,
                    verbose=verbose,
                    max_retries=max_retries,
                    ablation_mode=ablation_mode,  # ✅ 新增
                )


                if saver:
                    saver.save_result(result, key="dialogue_id")
                    saved = saver.get_completed_count()
                else:
                    saved = (i - start_index + 1)

                pbar.set_postfix({"saved": saved, "failed": failed, "skipped": skipped})

            except Exception as e:
                failed += 1
                if verbose:
                    print(f"[ERROR] dialogue_id={dialogue_id} failed: {e}")
                pbar.set_postfix({"failed": failed, "skipped": skipped})
            finally:
                pbar.update(1)

    if saver:
        with FileLock(saver.lock_file, timeout=10):
            if os.path.exists(saver.output_file):
                with open(saver.output_file, "r", encoding="utf-8") as f:
                    return json.load(f)
    return []


# ===================== CLI =====================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("CB history+principle simulation")
    parser.add_argument("--dev_path", type=str, default=DEFAULT_DEV_PATH)
    parser.add_argument("--persona_path", type=str, default=DEFAULT_PERSONA_PATH)
    parser.add_argument("--cluster_config", type=str, default=DEFAULT_CLUSTER_CONFIG)
    parser.add_argument("--max_turns", type=int, default=9)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--disable_incremental_save", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "w/o_depth", "w/o_breadth", "w/o_both", "w/o_expend"],
    )
    args = parser.parse_args()

    results = run_batch_evaluation(
        dev_path=args.dev_path,
        persona_path=args.persona_path,
        cluster_config=args.cluster_config,
        max_turns=args.max_turns,
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        output_file=args.output_file,
        enable_incremental_save=not args.disable_incremental_save,
        verbose=args.verbose,
        max_retries=args.max_retries,
        ablation_mode=args.ablation_mode,  # ✅ 新增
    )

    if args.disable_incremental_save:
        # 统一保存
        if args.output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = f"outputs/CB/evaluate/history/{args.cluster_config}/results_{ts}.json"
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output_file}")
    else:
        print(f"Saved incrementally to {args.output_file}")

    print(f"Total results: {len(results)}")
