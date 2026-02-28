#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ESC Principle-guided emotional-support simulation (history-based retrieval).

核心逻辑（ESC版）：
- 用对话 history embedding 检索最相似簇 / MCT 策略路径（retrieve_strategy_chain_by_history）；
- 在该簇内，基于最近一条 seeker utterance（get_last_utterance("seeker")）检索 top-k principles；
  * 检索 key = principle_text 的 WHEN 子句（逗号前）；
- 将 depth（策略链提示）+ breadth（principle->guidance / immediate）传给 Simulator 生成回复；
- 用 Seeker persona 生成 Seeker 的下一句（可替换为你的 seeker 生成器）；
- 用 ESC judge (A/B/C/D) 判断是否结束；
- 增量保存与断点续传：按 dialogue_id 去重。

数据假设：
- ESC dev/test 数据（用于跑模拟）格式（你自己对齐即可）：
[
  {
    "index": 0,
    "emotion_type": "...",
    "problem_type": "...",
    "situation": "...",
    "seeker_persona_used": "...",   # 可选
  },
  ...
]

输出 result（每条对话一次）：
{
  "dialogue_id": ...,
  "num_turns": ...,
  "final_attitude_prediction": "A|B|C|D",
  "dialog": [...],
  "meta": {...},
  "ablation_mode": ...
}
"""

import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from tqdm import tqdm
from filelock import FileLock

# =============== retrieval / embedding ===============
from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL
from src.utils.retrieve import retrieve_strategy_chain_by_history

# =============== LLM APIs ===============
from src.utils.llm_api import call_plato_api

# =============== guidance helpers (你已有的，CB里在用) ===============
from src.utils.guidance_esc import generate_high_level_guidance, generate_strategy_chain_summary  # ✅ 直接复用你已有的工具
# =============== 如果你已有类似 format_strategy_hint 的工具，就复用 ===============
from src.utils.llm_dialog import format_strategy_hint

# ===================== 路径配置（ESC版） =====================
DEFAULT_CLUSTER_CONFIG = "kmeans_k150"

CLUSTERS_PATH   = "outputs/ESC/cluster/history/{}/clusters.json"
TREES_DIR       = "outputs/ESC/cluster/history/{}/topk"
PRINCIPLES_DIR  = "outputs/ESC/cluster/history/{}/principles_by_cluster"

# =============== 模型配置 ===============
MODEL = os.getenv("GPT_MODEL", "gpt-3.5-turbo-0125")

# =============== 数据路径（你按需改） ===============
DEFAULT_DEV_PATH = os.getenv("ESC_DEV_PATH", "data/ESC/dev.json")
DEFAULT_PERSONA_PATH = os.getenv("ESC_PERSONA_PATH", "outputs/P4G/personas/personas_eval.jsonl")

# =============== 全局 Embedder 与缓存 ===============
_EMBEDDER: Optional[Embedder] = None
_CLUSTER_PRINCIPLES_CACHE: Dict[str, List[Dict[str, Any]]] = {}
_CLUSTER_PRINCIPLES_EMB_CACHE: Dict[str, List[List[float]]] = {}


# ===================== 基础工具 =====================

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


def load_json_auto(path: str) -> Any:
    """支持 JSON / JSONL"""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    if "\n" in content and not content.lstrip().startswith("["):
        data = []
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data
    return json.loads(content)


# ===================== principles 检索 =====================

def extract_when_clause(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    idx = text.find(",")
    return text if idx == -1 else text[:idx].strip()


def dot_similarity(v1: List[float], v2: List[float]) -> float:
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def load_principles_for_cluster(cluster_id: int, cluster_config: str = DEFAULT_CLUSTER_CONFIG) -> List[Dict[str, Any]]:
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
    _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = []  # lazy embedding
    return cleaned


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
    for turn in reversed(dialog_history):
        if (turn.get("speaker") or "").lower() == speaker.lower():
            txt = turn.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return ""
def history_to_plain_text_esc(dialog_history: List[Dict[str, Any]]) -> str:
    # 简单格式化：turn_id|speaker: text
    lines = []
    for t in dialog_history:
        tid = t.get("turn_id", 0)
        sp = t.get("speaker", "unknown")
        tx = (t.get("text") or "").strip()
        lines.append(f"turn_{tid}|{sp}: {tx}")
    return "\n".join(lines).strip()


def call_esc_judge(dialog_history: List[Dict[str, Any]], meta: Dict[str, Any], model: str, max_tokens: int = 16) -> str:
    dialogue_text = history_to_plain_text_esc(dialog_history)
    user_prompt = ESC_JUDGE_USER_TMPL.format(
        emotion_type=meta.get("emotion_type", ""),
        problem_type=meta.get("problem_type", ""),
        dialogue_text=dialogue_text,
    )
    messages = [{"role": "system", "content": ESC_JUDGE_SYSTEM},
                {"role": "user", "content": user_prompt}]
    print(f"\n[ESC Judge] dialogue_id={meta.get('dialogue_id','')} messages:\n{messages}\n")
    out = call_plato_api(messages, model=model, temperature=0.0, max_tokens=max_tokens)
    out = (out or "").strip()

    # 只取首行/首字母做稳健解析
    first = out.splitlines()[0].strip() if out else ""
    if first.startswith("A"):
        return "A"
    if first.startswith("B"):
        return "B"
    if first.startswith("C"):
        return "C"
    if first.startswith("D"):
        return "D"
    # 兜底：如果模型跑偏
    return "B"


# ===================== 增量保存（ESC：按 dialogue_id 去重） =====================

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


# ===================== ESC 生成（Simulator / Seeker） =====================

def chat_completion_simulator(messages, model: str, temperature: float, max_tokens: int) -> str:
    # 你也可以改成 openai compatible / siliconflow chat completions
    return call_plato_api(messages, model=model, temperature=temperature, max_tokens=max_tokens)

def chat_completion_seeker(messages, model: str, temperature: float, max_tokens: int) -> str:
    return call_plato_api(messages, model=model, temperature=temperature, max_tokens=max_tokens)


from typing import Optional, Tuple
from textwrap import dedent

def build_simulator_prompt(
    dialogue_text: Optional[str] = None,
    dialog_history: Optional[List[Dict[str, Any]]] = None,
    strategy_hint_text: Optional[str] = None,
    guidance_text: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Standard ESC Simulator prompt, baseline-consistent, but with two perspectives:
    - Long-term planning (strategy hint)
    - Immediate response (high-level guidance)
    """
    system_prompt = dedent("""
    Now enter the role-playing mode. You are the simulator who is trying to help the seeker(patient) reduce their emotional distress and help them understand and work through the challenges.
    """).strip()

    if dialogue_text is None and dialog_history is not None:
        dialogue_text = history_to_plain_text_esc(dialog_history)

    sections = [
        "[conversation]",
        dialogue_text.strip() if isinstance(dialogue_text, str) else "",
    ]

    # Perspective 1: Long-term planning
    if strategy_hint_text and str(strategy_hint_text).strip():
        sections.append(dedent(f"""
        [Long-term planning]
        {str(strategy_hint_text).strip()}
        """).strip())

    # Perspective 2: Immediate response
    if guidance_text and str(guidance_text).strip():
        sections.append(dedent(f"""
        [Immediate response guidance]
        {str(guidance_text).strip()}
        """).strip())

    # Instruction: keep baseline style (one short sentence) + explicitly use both views
    sections.append(dedent("""
    [Instruction]
    Think from two perspectives:
    1) Long-term planning: follow the long-term planning block to keep a coherent helping trajectory.
    2) Immediate response: follow the immediate guidance to address the seeker’s latest needs.

    Use both to decide your next move,don't just copy one of them.
    Please reply with only one short and succinct sentence for simulator.
    [simulator]:
    """).strip())

    user_prompt = "\n\n".join([s for s in sections if s])

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_seeker_prompt(
    dialog_history: List[Dict[str, Any]],
    meta: Dict[str, Any],
    seeker_persona: str,
) -> List[Dict[str, str]]:
    dialogue_text = history_to_plain_text_esc(dialog_history)
    sys = (
        "Now enter the role-playing mode. In the following conversation, you will play as a patient in a counselling conversation with a therapist."
    )
    usr = f"""
    You are the patient who is looking for the help from the simulator(therapist), because you have the emotional issue about {meta.get("emotion_type","")}regarding {meta.get("problem_type","")}.

    Important:
    - Stay consistent with your emotional issue (emotion_type) and problem context (problem_type).
    - If you think your are convinced by the simulator, you can say "solve the problem".

    The following is the conversation so far:
    [conversation]
    {dialogue_text}
    [/conversation]

    Please reply with only one short and succinct sentence.
    [seeker]:
    """
    return [{"role": "system", "content": sys},
            {"role": "user", "content": usr}]


def generate_simulator_utterance(
    dialog_history: List[Dict[str, Any]],
    meta: Dict[str, Any],
    seeker_persona: str,
    strategy_chain_hint: Optional[str],
    guidance_text: Optional[str],
    model: str,
    temperature: float = 0.2,
    max_tokens: int = 160,
) -> str:
    def _empty(x): return x is None or (isinstance(x, str) and x.strip() == "")
    use_depth = not _empty(strategy_chain_hint)
    use_breadth = not _empty(guidance_text)

    msgs = build_simulator_prompt(
        dialog_history=dialog_history,
        strategy_hint_text=strategy_chain_hint if use_depth else None,
        guidance_text=guidance_text if use_breadth else None,
    )
    print(f"\n[Simulator Prompt] dialogue_id={meta.get('dialogue_id','')} messages:\n{msgs}\n")
    out = chat_completion_simulator(msgs, model=model, temperature=temperature, max_tokens=max_tokens)
    return (out or "").strip()


def generate_seeker_utterance(
    dialog_history: List[Dict[str, Any]],
    meta: Dict[str, Any],
    seeker_persona: str,
    model: str,
    temperature: float = 0.5,
    max_tokens: int = 128,
) -> str:
    msgs = build_seeker_prompt(dialog_history, meta, seeker_persona)
    print(f"\n[Seeker Prompt] dialogue_id={meta.get('dialogue_id','')} messages:\n{msgs}\n")
    out = chat_completion_seeker(msgs, model=model, temperature=temperature, max_tokens=max_tokens)
    return (out or "").strip()


# ===================== 初始 turn0（ESC） =====================

def build_initial_turn0(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    # 你也可以改成 ESC 官方的初始两句模板
    return [
        {"turn_id": 0, "speaker": "simulator", "text": "Hello, how can I help you today?"},
        {"turn_id": 0, "speaker": "seeker", "text": (meta.get("situation") or "").strip()},
    ]


def count_rounds(dialog_history: List[Dict[str, Any]]) -> int:
    tids = []
    for t in dialog_history:
        try:
            tids.append(int(t.get("turn_id", -1)))
        except Exception:
            continue
    return (max(tids) + 1) if tids else 0


def get_first_strategy_group_from_retrieved(retrieved_chains: List[Any]) -> str:
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


# ===================== 主模拟流程（ESC版） =====================

def run_simulation_one_dialog(
    dialogue_id: str,
    meta: Dict[str, Any],
    seeker_persona: str,
    max_turns: int = 9,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
    verbose: bool = False,
    max_retries: int = 3,
    ablation_mode: str = "none",  # none | w/o_depth | w/o_breadth | w/o_both | w/o_expend
) -> Dict[str, Any]:

    dialog_history = build_initial_turn0(meta)
    current_turn_id = 1

    mode = (ablation_mode or "none").strip()
    valid_modes = {"none", "w/o_depth", "w/o_breadth", "w/o_both", "w/o_expend"}
    if mode not in valid_modes:
        mode = "none"

    disable_depth = mode in {"w/o_depth", "w/o_both"}
    disable_breadth = mode in {"w/o_breadth", "w/o_both"}
    use_first_group_as_immediate = (mode == "w/o_expend")

    final_label = "B"

    while True:
        clusters_path = CLUSTERS_PATH.format(cluster_config)
        trees_dir = TREES_DIR.format(cluster_config)

        # ---- 1) retrieve strategy chain (top-1) ----
        try:
            retrieved_chains = retrieve_strategy_chain_by_history(
                dialog_history=dialog_history,
                emb_fn=emb_fn,
                clusters_path=clusters_path,
                trees_dir=trees_dir,
                current_turn_id=current_turn_id,
                top_k=1,
            )
            print(f"[info] retrieved strategy chains: {retrieved_chains}")
        
        except Exception as e:
            print(f"[warn] retrieve_strategy_chain_by_history failed: {e}")
            retrieved_chains = []

        strategy_hint_text = format_strategy_hint(retrieved_chains, depth=3) or ""

        # ---- 2) cluster_id ----
        cluster_id = None
        if retrieved_chains:
            c0 = retrieved_chains[0]
            if isinstance(c0, dict):
                cluster_id = c0.get("cluster_id") or c0.get("cluster") or c0.get("cluster_idx")
                # ✅ 把字符串 cluster_id 尝试转 int，避免 principles 永远空
                if isinstance(cluster_id, str):
                    try:
                        cluster_id = int(cluster_id)
                    except Exception:
                        cluster_id = None

        # ---- 3) principles retrieval: query = last seeker utterance ----
        query_text = get_last_utterance(dialog_history, speaker="seeker")
        # print(f"[info] last seeker utterance for principle retrieval: '{query_text}'")
        top_principles: List[Dict[str, Any]] = []
        if isinstance(cluster_id, int):
            top_principles = get_top_principles_for_cluster(
                cluster_id=cluster_id,
                query_text=query_text,
                top_k=5,
                cluster_config=cluster_config,
            )

        # recent dialogue for summary/guidance
        recent_dialogue = history_to_plain_text_esc(dialog_history)

        # ---- breadth ----
        breadth_guidance_to_feed = ""
        if use_first_group_as_immediate:
            first_group = get_first_strategy_group_from_retrieved(retrieved_chains)
            breadth_guidance_to_feed = f"Immediate suggestion (use this now): {first_group}" if first_group else ""
        else:
            if not disable_breadth:
                breadth_guidance_to_feed = generate_high_level_guidance(
                    principles=top_principles,
                    last_user_utt=query_text,         # ✅ seeker last utt
                    recent_dialogue=recent_dialogue,  # ✅ 让 guidance 更贴近上下文
                    model=MODEL,
                    temperature=0.5,
                    max_tokens=120,
                )
                print(f"[info] breadth guidance generated: '{breadth_guidance_to_feed}'")
            else:
                breadth_guidance_to_feed = ""

        # ---- depth ----
        depth_hint_to_feed = ""
        used_strategy_hint = ""

        if disable_depth:
            depth_hint_to_feed = ""
            used_strategy_hint = ""
        else:
            strategy_chain_summary = generate_strategy_chain_summary(
                strategy_chain_hint=strategy_hint_text,
                recent_dialogue=recent_dialogue,
                model=MODEL,
                temperature=0.5,
                max_tokens=120,
            )
            depth_hint_to_feed = strategy_chain_summary
            used_strategy_hint = strategy_chain_summary
            print(f"[info] depth strategy hint generated: '{depth_hint_to_feed}'")

        # ---- 4) Simulator utterance ----
        sim_text = generate_simulator_utterance(
            dialog_history=dialog_history,
            meta=meta,
            seeker_persona=seeker_persona,
            strategy_chain_hint=depth_hint_to_feed,
            guidance_text=breadth_guidance_to_feed,
            model=MODEL,
            temperature=0.2,
            max_tokens=180,
        )

        sim_turn = {
            "turn_id": current_turn_id,
            "speaker": "simulator",
            "text": sim_text,
            "strategy_hint": retrieved_chains,
            "strategy_hint_text": used_strategy_hint,
            "principle_hint": top_principles,
            "high_level_guidance": breadth_guidance_to_feed,
            "ablation_mode": mode,
        }
        dialog_history.append(sim_turn)

        # ---- 5) Seeker reply ----
        seeker_text = generate_seeker_utterance(
            dialog_history=dialog_history,
            meta=meta,
            seeker_persona=seeker_persona,
            model=MODEL,
            temperature=0.6,
            max_tokens=128,
        )
        dialog_history.append({"turn_id": current_turn_id, "speaker": "seeker", "text": seeker_text})

        if verbose:
            print(f"\n[dialogue_id={dialogue_id}] turn={current_turn_id} mode={mode}")
            print("[simulator]:", sim_text)
            print("[seeker]:", seeker_text)
            print("------------------------------------------------")

        # ---- 6) judge ----
        from src.evaluate.critic_model_esc import call_critic_model  # ✅ 直接复用你已有的 ESC critic 调用（带重试）
        attitude, should_end, reward, critic_attitudes = call_critic_model(
            dialog_history,
            meta=meta,
            max_retries=max_retries,
            return_details=True
        )

        dialog_history[-1]["attitude"] = attitude
        dialog_history[-1]["reward"] = reward
        dialog_history[-1]["critic_attitudes"] = critic_attitudes

        if should_end or current_turn_id >= max_turns:
            if verbose:
                print(f"[ESC Critic] attitude={attitude}, reward={reward:.3f}, should_end={should_end}")
            break


        current_turn_id += 1

    return {
        "dialogue_id": dialogue_id,
        "num_turns": count_rounds(dialog_history),
        "final_attitude_prediction": final_label,
        "dialog": dialog_history,
        "meta": meta,
        "ablation_mode": mode,
    }


# ===================== Batch evaluation（ESC版） =====================

def load_esc_dev_dataset(dev_path: str) -> List[Dict[str, Any]]:
    data = load_json_auto(dev_path)
    if not isinstance(data, list):
        raise ValueError("ESC dev.json top-level must be a list")
    return data


def load_esc_personas(persona_path: str) -> List[str]:
    personas = load_json_auto(persona_path)
    if not isinstance(personas, list):
        raise ValueError("ESC persona file top-level must be a list (jsonl -> list)")
    out: List[str] = []
    for p in personas:
        if isinstance(p, dict):
            out.append(p.get("description") or p.get("persona") or p.get("seeker_persona_used") or "")
        else:
            out.append(str(p))
    return out


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
    ablation_mode: str = "none",
) -> List[Dict[str, Any]]:

    dev_data = load_esc_dev_dataset(dev_path)
    persona_list = load_esc_personas(persona_path)

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
        output_file = f"outputs/ESC/evaluate/history/{cluster_config}/results_{safe_mode}_{ts}.json"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    saver = IncrementalSaver(output_file) if enable_incremental_save else None

    skipped = 0
    failed = 0

    indices = []
    for i in range(start_index, end_index + 1):
        sample = dev_data[i]
        dialogue_id = sample.get("dialogue_id") or sample.get("id") or sample.get("index") or f"esc_{i:06d}"
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

    with tqdm(total=total, desc="ESC history+principle", unit="dlg", initial=skipped) as pbar:
        for i in indices:
            sample = dev_data[i]
            seeker_persona = persona_list[i]
            dialogue_id = sample.get("dialogue_id") or sample.get("id") or sample.get("index") or f"esc_{i:06d}"

            meta = {
                "emotion_type": sample.get("emotion_type", ""),
                "problem_type": sample.get("problem_type", ""),
                "situation": sample.get("situation", ""),
            }

            # 如果 dev 里自带 seeker_persona_used，就优先用它（比外部 persona 更稳）
            if isinstance(sample.get("seeker_persona_used"), str) and sample["seeker_persona_used"].strip():
                seeker_persona = sample["seeker_persona_used"].strip()

            try:
                result = run_simulation_one_dialog(
                    dialogue_id=str(dialogue_id),
                    meta=meta,
                    seeker_persona=seeker_persona,
                    max_turns=max_turns,
                    cluster_config=cluster_config,
                    verbose=verbose,
                    max_retries=max_retries,
                    ablation_mode=ablation_mode,
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

    parser = argparse.ArgumentParser("ESC history+principle simulation")
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
        ablation_mode=args.ablation_mode,
    )

    if args.disable_incremental_save:
        if args.output_file is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            args.output_file = f"outputs/ESC/evaluate/history/{args.cluster_config}/results_{ts}.json"
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Saved to {args.output_file}")
    else:
        print(f"Saved incrementally to {args.output_file}")

    print(f"Total results: {len(results)}")