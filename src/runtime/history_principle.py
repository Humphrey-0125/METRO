#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Principle-guided persuasion dialogue simulation.

核心逻辑：
- 先用对话历史检索最相似的簇 / MCT 策略路径（和原代码一致）；
- 再在该簇中，基于“最近一条 Persuadee 的 utterance”检索 top-5 micro-principles；
  * 检索用的是 principles 的 WHEN 子句（逗号前部分）；
  * principle_text 完整句子仍用于给 LLM 看；
- 将策略链提示 + principle 提示一起传给 Persuader 生成回复。
"""

import os
import json
import random
from typing import List, Dict, Any, Optional
from textwrap import dedent
from datetime import datetime
from tqdm import tqdm
from filelock import FileLock
from src.evaluate.critic_model import call_critic_model  # 对话结束判断和态度评估
from src.utils.llm_dialog import (
    format_strategy_hint,
    generate_persuader_utterance,
    generate_persuadee_utterance,
)

# embedding & 检索
from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL
from src.utils.retrieve import retrieve_strategy_chain_by_history

from src.utils.guidance import generate_high_level_guidance, generate_strategy_chain_summary
# ===================== 路径配置 =====================

# 默认聚类配置
DEFAULT_CLUSTER_CONFIG = "kmeans_k150"

# 动态路径：根据聚类配置生成
CLUSTERS_PATH = "outputs/P4G/cluster/{}/clusters.json"
STATES_PATH   = "outputs/P4G/embedding/history/states_embeddings_train_test.json"
TREES_DIR     = "outputs/P4G/cluster/{}/topk"
PRINCIPLES_DIR = "outputs/P4G/cluster/{}/principles_by_cluster"

# 按簇保存的 principles 路径（由 split_principles_by_cluster.py 生成）
PERSUADER_GPT_MODEL = "gpt-3.5-turbo"
PERSUADEE_GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL="gpt-3.5-turbo"  

# ===================== 全局 Embedder 与缓存 =====================

_EMBEDDER: Optional[Embedder] = None

# 缓存：cluster_id -> [ {id, principle_type, principle_text}, ... ]
_CLUSTER_PRINCIPLES_CACHE: Dict[int, List[Dict[str, Any]]] = {}
# 缓存：cluster_id -> [ embedding_of_when_clause, ... ]
_CLUSTER_PRINCIPLES_EMB_CACHE: Dict[int, List[List[float]]] = {}


def get_embedder() -> Embedder:
    """懒加载全局 Embedder，避免重复初始化。"""
    global _EMBEDDER
    if _EMBEDDER is None:
        _EMBEDDER = Embedder(
            model_name=EMBED_MODEL_NAME,
            api_key=EMBED_API_KEY,
            api_url=EMBED_API_URL,
        )
    return _EMBEDDER


def emb_fn(text: str) -> List[float]:
    """封装好的 embedding 函数，底层用 src.utils.embed.Embedder.encode。"""
    embedder = get_embedder()
    return embedder.encode(text or "")


# ===================== principles 相关工具函数 =====================

def extract_when_clause(text: str) -> str:
    """
    从 principle_text 中提取用于检索的 WHEN 子句：
    - 原始格式一般为: "When ..., you should ..." 或 "When ..., avoid ..."
    - 我们只取第一个逗号之前的部分作为“用户状态描述”。
    """
    if not text:
        return ""
    text = text.strip()
    idx = text.find(",")
    if idx == -1:
        return text
    return text[:idx].strip()


def load_principles_for_cluster(cluster_id: int, cluster_config: str = DEFAULT_CLUSTER_CONFIG) -> List[Dict[str, Any]]:
    """
    读取某个簇的所有 principles（带缓存）。
    文件路径：PRINCIPLES_DIR.format(cluster_config)/cluster_{cluster_id}.json
    """
    # 使用 cluster_config 作为缓存键的一部分，避免不同配置之间的缓存冲突
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
    _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = []  # 延迟计算 embedding
    return cleaned


def dot_similarity(v1: List[float], v2: List[float]) -> float:
    """
    点积相似度。
    由于 Embedder.encode 已经做了 L2 归一化，这里 dot product 就是 cosine similarity。
    """
    if not v1 or not v2 or len(v1) != len(v2):
        return 0.0
    return sum(a * b for a, b in zip(v1, v2))


def get_top_principles_for_cluster(
    cluster_id: int,
    query_text: str,
    top_k: int = 5,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
) -> List[Dict[str, Any]]:
    """
    在给定簇中，用 query_text 的 embedding 检索最相似的 top_k 条 principles。
    检索 key = principles 的 WHEN 子句（逗号前）。
    """
    if not query_text:
        return []

    principles = load_principles_for_cluster(cluster_id, cluster_config)
    if not principles:
        return []

    # 使用 cluster_config 作为缓存键的一部分
    cache_key = f"{cluster_config}_{cluster_id}"
    
    # 计算 / 读取该簇内 principles 的 embedding（WHEN 子句）
    if not _CLUSTER_PRINCIPLES_EMB_CACHE.get(cache_key, []):
        emb_list: List[List[float]] = []
        for p in principles:
            full_txt = p.get("principle_text") or ""
            when_clause = extract_when_clause(full_txt)
            emb_list.append(emb_fn(when_clause))
        _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = emb_list

    query_emb = emb_fn(query_text)
    cand_embs = _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key]

    sims = []
    for idx, emb in enumerate(cand_embs):
        sim = dot_similarity(query_emb, emb)
        sims.append((sim, idx))

    sims.sort(key=lambda x: x[0], reverse=True)
    top = sims[:top_k]

    results: List[Dict[str, Any]] = []
    for sim, idx in top:
        item = principles[idx].copy()
        item["similarity"] = sim  # 方便调试，可后续丢弃
        results.append(item)

    return results


def get_last_persuadee_utterance(dialog_history: List[Dict[str, Any]]) -> str:
    """
    从当前对话历史中，找到最近一条 Persuadee 的 text。
    找不到则返回空字符串。
    """
    for turn in reversed(dialog_history):
        if turn.get("speaker") == "Persuadee":
            txt = turn.get("text") or ""
            if isinstance(txt, str) and txt.strip():
                return txt.strip()
    return ""

def get_first_strategy_group_from_retrieved(retrieved_chains: List[Any]) -> Optional[str]:
    """
    从 retrieve_strategy_chain_by_history 的返回（top-1）里取 chain 的第一层策略组。
    返回格式：
      - 单个策略: 'ask_question'
      - 多个策略: 'ask_question | build_rapport'
    """
    if not retrieved_chains:
        return None
    c0 = retrieved_chains[0]
    if not isinstance(c0, dict):
        return None

    chain = c0.get("chain")
    if not isinstance(chain, list) or len(chain) == 0:
        return None

    first_group = chain[0]
    if not isinstance(first_group, list) or len(first_group) == 0:
        return None

    parts = [str(x).strip() for x in first_group if str(x).strip()]
    return " | ".join(parts) if parts else None

# ===================== 数据加载 =====================
from src.utils.load import load_dev_dataset
from src.utils.load import load_personas

# ===================== 增量保存机制 =====================
class IncrementalSaver:
    """
    跨进程安全的增量保存器，支持实时保存结果到 JSON 文件
    使用文件锁保证多个进程同时写入时的安全性
    """
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.lock_file = output_file + ".lock"
        self.completed_indices = set()  # 记录已完成的索引（persona_index），用于断点续传
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 如果文件已存在，加载已有结果
        if os.path.exists(output_file):
            try:
                with FileLock(self.lock_file, timeout=10):
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if isinstance(existing_results, list):
                            # 提取已完成的 persona_index
                            for result in existing_results:
                                if "persona_index" in result:
                                    self.completed_indices.add(result["persona_index"])
                            print(f"[断点续传] 已加载 {len(existing_results)} 个已有结果，{len(self.completed_indices)} 个 persona 已完成")
            except Exception as e:
                print(f"[警告] 加载已有文件失败: {e}，将重新开始")
                self.completed_indices = set()
    
    def save_result(self, result: Dict[str, Any]):
        """
        跨进程安全地保存单个结果（使用文件锁）
        """
        # 使用文件锁保证跨进程安全
        with FileLock(self.lock_file, timeout=30):
            # 读取现有结果
            existing_results = []
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if not isinstance(existing_results, list):
                            existing_results = []
                except Exception as e:
                    print(f"[警告] 读取现有结果失败: {e}，将重新开始")
                    existing_results = []
            
            # 检查是否已存在（避免重复添加）
            persona_index = result.get("persona_index")
            if persona_index is not None:
                # 检查是否已经存在相同 persona_index 的结果
                already_exists = any(
                    r.get("persona_index") == persona_index 
                    for r in existing_results
                )
                if already_exists:
                    # 如果已存在，替换它
                    existing_results = [
                        r for r in existing_results 
                        if r.get("persona_index") != persona_index
                    ]
                    existing_results.append(result)
                else:
                    # 如果不存在，直接添加
                    existing_results.append(result)
                    self.completed_indices.add(persona_index)
            else:
                # 如果没有 persona_index，直接添加
                existing_results.append(result)
            
            # 写入文件（使用临时文件确保原子性）
            temp_file = self.output_file + ".tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                # 原子性替换
                os.replace(temp_file, self.output_file)
            except Exception as e:
                print(f"[错误] 保存结果失败: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
    
    def is_completed(self, index: int) -> bool:
        """
        检查指定索引（persona_index）的任务是否已完成
        需要读取文件来检查（因为不同进程的内存状态不同）
        """
        # 使用文件锁读取文件检查
        with FileLock(self.lock_file, timeout=10):
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if isinstance(existing_results, list):
                            # 检查是否存在该索引
                            return any(
                                r.get("persona_index") == index 
                                for r in existing_results
                            )
                except Exception:
                    pass
        return index in self.completed_indices
    
    def get_completed_count(self) -> int:
        """
        获取已完成的任务数量（从文件读取）
        """
        with FileLock(self.lock_file, timeout=10):
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if isinstance(existing_results, list):
                            return len(existing_results)
                except Exception:
                    pass
        return 0

# ===================== 主模拟流程 =====================

# Prompt type 配置
PERSUADER_PROMPT_TYPE = "Ours_1"   # 用于 Persuader，默认使用 runtime_template
CRITIC_PROMPT_TYPE = "LDPP"  # 用于 Persuadee 和 Critic
PERSUADEE_PROMPT_TYPE = "personas"  # 用于 Persuadee，使用 resistance 描述

def run_simulation_with_initial(
    initial_dialog: List[Dict[str, Any]],
    max_turns: int = 9,
    persuader_prompt_type: Optional[str] = None,
    persuadee_prompt_type: Optional[str] = None,
    critic_prompt_type: Optional[str] = None,
    persona_description: Optional[str] = None,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
    verbose: bool = True,
    max_retries: int = 3,
    ablation_mode: str = "none",  # "none" | "w/o_depth" | "w/o_breadth" | "w/o_both" | "w/o_expend"
) -> List[Dict[str, Any]]:
    """
    使用初始对话运行模拟（加入 principles 检索），支持消融：

    depth   = 策略链（strategy_chain_hint / summary）
    breadth = principles -> high_level_guidance（展开/扩展）

    Modes:
    - none:        depth + breadth
    - w/o_depth:   breadth only（不喂策略链 / 不做 summary）
    - w/o_breadth: depth only（不生成 high_level_guidance / 不喂 guidance）
    - w/o_both:    两者都不喂
    - w/o_expend:  仍然两视角，但不做 breadth 扩展：
                  Immediate 不用 high_level_guidance，而是用策略链第一层策略组（retrieved_chains[0]["chain"][0]）
    """
    if persuader_prompt_type is None:
        persuader_prompt_type = PERSUADER_PROMPT_TYPE
    if persuadee_prompt_type is None:
        persuadee_prompt_type = PERSUADEE_PROMPT_TYPE
    if critic_prompt_type is None:
        critic_prompt_type = CRITIC_PROMPT_TYPE

    # --------- ablation flags ----------
    import time
    mode = (ablation_mode or "none").strip()
    valid_modes = {"none", "w/o_depth", "w/o_breadth", "w/o_both", "w/o_expend"}
    if mode not in valid_modes:
        mode = "none"

    disable_depth = mode in {"w/o_depth", "w/o_both"}
    disable_breadth = mode in {"w/o_breadth", "w/o_both"}
    use_first_group_as_immediate = (mode == "w/o_expend")

    print(f"ablation_mode: {mode}")

    # 复制初始对话
    dialog_history: List[Dict[str, Any]] = list(initial_dialog)

    # 设置下一轮 turn_id
    current_turn_id = max((turn.get("turn_id", 0) for turn in dialog_history), default=0) + 1

    while True:
        # 1) 检索策略链（top-1）
        clusters_path = CLUSTERS_PATH.format(cluster_config)
        trees_dir = TREES_DIR.format(cluster_config)

        try:
            retrieved_chains = retrieve_strategy_chain_by_history(
                dialog_history=dialog_history,
                emb_fn=emb_fn,
                clusters_path=clusters_path,
                trees_dir=trees_dir,
                current_turn_id=current_turn_id,
                top_k=1,
            )
        except Exception as e:
            print(f"[warn] retrieve_strategy_chain_by_history failed: {e}")
            retrieved_chains = []

        # print(f"retrieved_chains: {retrieved_chains}")

        strategy_hint_text = format_strategy_hint(retrieved_chains,depth=3)
        print(f"strategy_hint_text: {strategy_hint_text}")

        # 2) principles 检索（top-5）
        cluster_id = None
        if retrieved_chains:
            c0 = retrieved_chains[0]
            if isinstance(c0, dict):
                cluster_id = c0.get("cluster_id") or c0.get("cluster") or c0.get("cluster_idx")

        top_principles: List[Dict[str, Any]] = []
        query_text = get_last_persuadee_utterance(dialog_history)  # 始终获取，用于 breadth 或 immediate

        if isinstance(cluster_id, int):
            top_principles = get_top_principles_for_cluster(
                cluster_id=cluster_id,
                query_text=query_text,
                top_k=5,
                cluster_config=cluster_config,
            )

        time2=time.time()

        # 3) 生成 breadth / immediate
        high_level_guidance: Optional[str] = None

        if use_first_group_as_immediate:
            # ✅ w/o_expend：不调用 generate_high_level_guidance
            first_group = get_first_strategy_group_from_retrieved(retrieved_chains)
            if first_group:
                # 仍然填到 guidance_text，让 prompt 维持“两视角”
                high_level_guidance = f"Immediate suggestion (use this now): {first_group}"
            else:
                high_level_guidance = None

        else:
            # 原逻辑：breadth 来自 principles → guidance（除非 w/o_breadth/w/o_both）
            if not disable_breadth:
                high_level_guidance = generate_high_level_guidance(
                    principles=top_principles,
                    last_user_utt=query_text,
                    model=PERSUADER_GPT_MODEL,
                    temperature=0.5,
                    max_tokens=100
                )
            else:
                high_level_guidance = None

        # 4) 生成 depth（raw 或 summary）
        used_strategy_hint_text: Optional[str] = None

        if disable_depth:
            used_strategy_hint_text = None
        else:
            if (persuader_prompt_type or "").strip() == "Ours_1":
                # Ours_1: 总结策略链（只有 depth enable 才调用）
                from src.utils.llm_dialog import history_to_plain_text
                if len(dialog_history) >= 2:
                    recent_dialogue = history_to_plain_text(dialog_history[-2:])
                elif len(dialog_history) >= 1:
                    recent_dialogue = history_to_plain_text(dialog_history[-1:])
                else:
                    recent_dialogue = ""

                used_strategy_hint_text = generate_strategy_chain_summary(
                    strategy_chain_hint=strategy_hint_text,
                    current_turn_id=current_turn_id,
                    recent_dialogue=recent_dialogue,
                    model=PERSUADER_GPT_MODEL,
                    temperature=0.5,
                    max_tokens=100
                )
            else:
                used_strategy_hint_text = strategy_hint_text

        time3=time.time()
        print(f"生成 guidance 和 summary 耗时: {time3-time2:.2f} 秒")

        print(f"used_strategy_hint_text: {used_strategy_hint_text}")
        print(f"high_level_guidance: {high_level_guidance}")
        # 5) 生成 Persuader 发言
        persuader_text = generate_persuader_utterance(
            dialog_history=dialog_history,
            strategy_chain_hint=used_strategy_hint_text,  # None => depth ablated
            guidance_text=high_level_guidance,            # None => breadth ablated; w/o_expend => first-group immediate
            prompt_type=persuader_prompt_type,
            model=PERSUADER_GPT_MODEL
        )

        time4=time.time()
        print(f"生成 Persuader 回复耗时: {time4-time3:.2f} 秒")

        if verbose:
            print(f"[persuader] (mode={mode}): {persuader_text}")
            print("--------------------------------")

        persuader_turn = {
            "turn_id": current_turn_id,
            "speaker": "Persuader",
            "text": persuader_text,
            "strategy_hint": retrieved_chains,
            "strategy_hint_text": used_strategy_hint_text,  # 实际喂给模型的 depth（summary/原始/None）
            "principle_hint": top_principles,
            "high_level_guidance": high_level_guidance,     # 实际喂给模型的 immediate/breadth（或 None）
            "ablation_mode": mode,
        }
        dialog_history.append(persuader_turn)

        # 6) Persuadee 回复
        persuadee_text = generate_persuadee_utterance(
            dialog_history=dialog_history,
            prompt_type=persuadee_prompt_type,
            persona_description=persona_description,
            model=PERSUADEE_GPT_MODEL
        )
        if verbose:
            print("[persuadee]: ", persuadee_text)
            print("================================================")

        persuadee_turn = {
            "turn_id": current_turn_id,
            "speaker": "Persuadee",
            "text": persuadee_text
        }
        dialog_history.append(persuadee_turn)

        # 7) critic 评估
        attitude, should_end, reward, critic_attitudes = call_critic_model(
            dialog_history,
            prompt_type=critic_prompt_type,
            max_retries=max_retries,
            return_details=True
        )
        dialog_history[-1]["attitude"] = attitude
        dialog_history[-1]["reward"] = reward
        dialog_history[-1]["critic_attitudes"] = critic_attitudes

        if should_end or current_turn_id >= max_turns:
            if verbose:
                print(f"Persuadee 态度评估为 {attitude}，模型建议结束对话")
            break

        current_turn_id += 1

    return dialog_history


import random


def process_single_evaluation(
    i: int,
    dataset: List[Dict[str, Any]],
    persona_list: List[str],
    max_turns: int,
    persuader_prompt_type: Optional[str],
    critic_prompt_type: Optional[str],
    persuadee_prompt_type: Optional[str],
    cluster_config: str,
    total_runs: int,
    verbose: bool = False,
    max_retries: int = 3,
    ablation_mode: str = "none",
) -> Optional[Dict[str, Any]]:
    """
    处理单个评估任务（单个对话）。

    Args:
        i: 当前任务索引
        dataset: 数据集
        persona_list: 人格列表
        max_turns: 最大轮数
        persuader_prompt_type: Persuader prompt 类型
        critic_prompt_type: Critic prompt 类型
        persuadee_prompt_type: Persuadee prompt 类型
        cluster_config: 聚类配置
        total_runs: 总任务数
        verbose: 是否显示详细输出

    Returns:
        评估结果字典，如果失败则返回 None
    """
    try:
        # 1) 随机从 dev 中选一条对话
        sample = random.choice(dataset)
        base_index = sample.get("index", f"sample_{i}")
        dialog = sample.get("dialog", [])

        # 2) 当前 persona（依次遍历所有人格）
        persona_idx = i  # 0..num_personas-1
        persona_description = persona_list[persona_idx]

        # 3) 组合一个可区分的数据索引（同一 dev + 不同 persona）
        index = f"{base_index}_persona{persona_idx}"

        # 提取前两轮对话作为初始化
        if len(dialog) >= 2:
            initial_turns = dialog[:2]
        elif len(dialog) >= 1:
            initial_turns = dialog[:1]
        else:
            return None

        # 运行模拟对话（并行执行时关闭详细输出，避免输出混乱）
        simulated_dialog = run_simulation_with_initial(
            initial_dialog=initial_turns,
            max_turns=max_turns,
            persuader_prompt_type=persuader_prompt_type,
            critic_prompt_type=critic_prompt_type,
            persuadee_prompt_type=persuadee_prompt_type,
            persona_description=persona_description,
            cluster_config=cluster_config,
            verbose=verbose,
            ablation_mode=ablation_mode,
        )

        result = {
            "original_index": index,
            "dev_index": base_index,
            "persona_used": persona_description,
            "persona_index": persona_idx,
            "initial_turns": len(initial_turns),
            "simulated_dialog": simulated_dialog,
            "total_turns": len(simulated_dialog) / 2,  # 每轮包括 P+Q 两条
            "generated_turns": (len(simulated_dialog) - len(initial_turns)) / 2,
        }
        return result

    except Exception as e:
        if verbose:
            print(f"\n[错误] 处理任务 {i+1}/{total_runs} 时出错: {e}")
        return None


def run_batch_evaluation(
    max_turns: int = 9,
    max_samples: Optional[int] = None,
    persuader_prompt_type: Optional[str] = None,
    critic_prompt_type: Optional[str] = None,
    persuadee_prompt_type: Optional[str] = None,
    cluster_config: str = DEFAULT_CLUSTER_CONFIG,
    verbose: bool = False,
    start_index: Optional[int] = None,
    end_index: Optional[int] = None,
    output_file: Optional[str] = None,
    enable_incremental_save: bool = True,
    ablation_mode: str = "none",
) -> List[Dict[str, Any]]:
    """
    对 dev.json 中的所有对话进行批量评估（串行版本，多进程并发由 shell 脚本控制）。
    每个对话依次选择第 1, 3, 5, ... 条人格（下标 0,2,4,...)，
    并在整个对话中模拟该 persona 的回复模式。

    Args:
        max_turns: 每个对话的最大轮数
        max_samples: 最大测试样本数（用于调试）
        persuader_prompt_type: Persuader 使用的 prompt 类型
        critic_prompt_type: Critic 使用的 prompt 类型
        persuadee_prompt_type: Persuadee 使用的 prompt 类型
        cluster_config: 聚类配置名称
        verbose: 是否显示详细输出（默认 False）
        start_index: 起始索引（用于批次处理，None 表示从 0 开始）
        end_index: 结束索引（用于批次处理，None 表示处理到 max_samples）
        output_file: 输出文件路径（如果启用增量保存）
        enable_incremental_save: 是否启用增量保存（默认 True）

    Returns:
        List[Dict[str, Any]]: 所有对话的评估结果
    """
    print("加载数据集...")
    dataset = load_dev_dataset()
    personas = load_personas()   # ← 加载所有 persona（pid -> description）
    persona_list = list(personas.values())

    if not dataset:
        print("无法加载数据集")
        return []

    if not persona_list:
        print("未加载到任何 persona，退出")
        return []

    num_personas = len(persona_list)  # e.g., 40
    # 每次评估：依次使用所有人格，各随机配一条 dev 样本
    if max_samples is None:
        total_runs = num_personas
    else:
        total_runs = min(max_samples, num_personas)

    # 确定实际处理的索引范围
    if start_index is not None:
        actual_start = start_index
    else:
        actual_start = 0
    
    if end_index is not None:
        actual_end = min(end_index, total_runs - 1)
    else:
        actual_end = total_runs - 1

    actual_total = actual_end - actual_start + 1

    # 初始化增量保存器
    saver = None
    if enable_incremental_save and output_file:
        saver = IncrementalSaver(output_file)
        completed_count = saver.get_completed_count()
        if completed_count > 0:
            print(f"[断点续传] 发现 {completed_count} 个已完成的任务，将跳过这些任务")
    else:
        print("[提示] 未启用增量保存，结果将在所有任务完成后统一保存")

    print(f"开始批量评估，索引范围: {actual_start} 到 {actual_end}，共 {actual_total} 轮（每轮对应一个 persona，dev 中随机采样一条对话）...")
    print(f"注意：多进程并发由 shell 脚本控制，本进程串行处理分配的任务")

    results: List[Dict[str, Any]] = []
    failed_count = 0
    skipped_count = 0

    # 串行处理所有任务（多进程并发由 shell 脚本层面控制）
    task_indices = []
    for i in range(actual_start, actual_end + 1):
        # 如果启用增量保存且该任务已完成，跳过
        if saver and saver.is_completed(i):
            skipped_count += 1
            continue
        task_indices.append(i)

    # 如果所有任务都已完成，直接返回已有结果
    if len(task_indices) == 0:
        print(f"[提示] 所有任务均已完成，共 {skipped_count} 个任务")
        if saver:
            # 从文件读取所有结果
            with FileLock(saver.lock_file, timeout=10):
                if os.path.exists(saver.output_file):
                    try:
                        with open(saver.output_file, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
        return []

    # 使用 tqdm 显示进度
    with tqdm(total=actual_total, desc="处理进度", unit="对话", initial=skipped_count) as pbar:
        for i in task_indices:
            try:
                result = process_single_evaluation(
                    i=i,
                    dataset=dataset,
                    persona_list=persona_list,
                    max_turns=max_turns,
                    persuader_prompt_type=persuader_prompt_type,
                    critic_prompt_type=critic_prompt_type,
                    persuadee_prompt_type=persuadee_prompt_type,
                    cluster_config=cluster_config,
                    total_runs=total_runs,
                    verbose=verbose,
                    ablation_mode=ablation_mode,      # ✅ 真正传到位
                )

                if result is not None:
                    results.append(result)
                    # 增量保存（使用文件锁保证跨进程安全）
                    if saver:
                        saver.save_result(result)
                    # 显示进度：使用保存器的总数（包括已有的和新完成的）
                    total_success = saver.get_completed_count() if saver else len(results)
                    pbar.set_postfix({
                        "成功": total_success,
                        "失败": failed_count,
                        "跳过": skipped_count
                    })
                else:
                    failed_count += 1
                    total_success = saver.get_completed_count() if saver else len(results)
                    pbar.set_postfix({
                        "成功": total_success,
                        "失败": failed_count,
                        "跳过": skipped_count
                    })
            except Exception as e:
                failed_count += 1
                if verbose:
                    print(f"\n[异常] 任务 {i+1} 执行异常: {e}")
                total_success = saver.get_completed_count() if saver else len(results)
                pbar.set_postfix({
                    "成功": total_success,
                    "失败": failed_count,
                    "跳过": skipped_count
                })
            finally:
                pbar.update(1)

    # 从文件读取所有结果（因为可能有其他进程也在写入）
    if saver:
        with FileLock(saver.lock_file, timeout=10):
            if os.path.exists(saver.output_file):
                try:
                    with open(saver.output_file, "r", encoding="utf-8") as f:
                        all_results = json.load(f)
                except Exception:
                    all_results = results
            else:
                all_results = results
    else:
        all_results = results

    print(f"\n批量评估完成，共处理 {len(all_results)} 个对话（成功），{failed_count} 个失败，{skipped_count} 个跳过")
    return all_results



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Principle-guided persuasion dialogue simulation (history-based).")
    parser.add_argument("--max_turns", type=int, default=9)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument(
        "--cluster_config",
        type=str,
        default=DEFAULT_CLUSTER_CONFIG,
        help="聚类配置名称，例如：kmeans_k50, kmeans_k100, kmeans_k150, kmeans_k200, kmeans_k250",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="是否显示详细输出（并行执行时建议关闭）",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="起始索引（用于批次处理，None 表示从 0 开始）",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="结束索引（用于批次处理，None 表示处理到 max_samples）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径（如果启用增量保存，None 表示自动生成）",
    )
    parser.add_argument(
        "--disable_incremental_save",
        action="store_true",
        help="禁用增量保存（默认启用）",
    )
    parser.add_argument(
        "--ablation_mode",
        type=str,
        default="none",
        choices=["none", "w/o_depth", "w/o_breadth", "w/o_both", "w/o_expend"],  # ✅ NEW
        help=(
            "Ablation mode:\n"
            "  none        : depth + breadth\n"
            "  w/o_depth   : breadth only\n"
            "  w/o_breadth : depth only\n"
            "  w/o_both    : neither\n"
            "  w/o_expend  : keep 2-view, but replace high_level_guidance with first strategy group\n"
        ),
    )

    args = parser.parse_args()

    # 确定输出文件路径
    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"outputs/P4G/evaluate/history/{args.cluster_config}/{args.ablation_mode}/120_metrics/results_{PERSUADER_PROMPT_TYPE}_{PERSUADEE_PROMPT_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    enable_incremental_save = not args.disable_incremental_save

    # 运行批量评估
    results = run_batch_evaluation(
        max_turns=args.max_turns,
        max_samples=args.max_samples,
        persuader_prompt_type=PERSUADER_PROMPT_TYPE,
        critic_prompt_type=CRITIC_PROMPT_TYPE,
        persuadee_prompt_type=PERSUADEE_PROMPT_TYPE,
        cluster_config=args.cluster_config,
        verbose=args.verbose,
        start_index=args.start_index,
        end_index=args.end_index,
        output_file=output_file,
        enable_incremental_save=enable_incremental_save,
        ablation_mode=args.ablation_mode,
    )

    # 如果未启用增量保存，则在这里统一保存
    if not enable_incremental_save:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"批量评估结果已保存到 {output_file}")
    else:
        print(f"批量评估结果已实时保存到 {output_file}")
    
    print(f"共评估了 {len(results)} 个对话")

'''
使用方式：

1. 进程级并发（推荐，类似 run_ours.sh）：
   bash run_history_principle.sh
   
   这种方式通过启动多个 Python 进程来加速，每个进程处理一部分样本。
   所有进程共享同一个输出文件，通过增量保存机制保证线程安全。

2. 单进程多线程（原有方式）：
   python src/runtime/history_principle.py --max_samples 2 --cluster_config kmeans_k150
   
3. 批次处理（手动指定范围）：
   python src/runtime/history_principle.py \
       --max_samples 200 \
       --cluster_config kmeans_k150 \
       --start_index 0 \
       --end_index 49 \
       --output_file outputs/P4G/evaluate/history/kmeans_k150/new_metrics/results.json

4. 自定义并行度：
   python src/runtime/history_principle.py --max_samples 40 --cluster_config kmeans_k150 --max_workers 1 --verbose

5. 显示详细输出（调试用）：
   python src/runtime/history_principle.py --max_samples 200 --cluster_config kmeans_k150 --verbose
'''
