#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Principle-guided persuasion dialogue simulation.

核心逻辑：
- 先用对话历史检索最相似的簇 / MCT 策略路径（和原代码一致）；
- 再在该簇中，基于"最近一条 Persuadee 的 utterance"检索 top-5 micro-principles；
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
from src.evaluate.critic_model import call_critic_model
from src.utils.llm_dialog import (
    format_strategy_hint,
    generate_persuader_utterance,
    generate_persuadee_utterance,
)

from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL
from src.utils.retrieve import retrieve_strategy_chain_by_history

from src.utils.guidance import generate_high_level_guidance, generate_strategy_chain_summary

DEFAULT_CLUSTER_CONFIG = "kmeans_k150"

CLUSTERS_PATH = "outputs/P4G/cluster/{}/clusters.json"
STATES_PATH   = "outputs/P4G/embedding/history/states_embeddings_train_test.json"
TREES_DIR     = "outputs/P4G/cluster/{}/topk"
PRINCIPLES_DIR = "outputs/P4G/cluster/{}/principles_by_cluster"

PERSUADER_GPT_MODEL = "gpt-3.5-turbo"
PERSUADEE_GPT_MODEL = "gpt-3.5-turbo"
GPT_MODEL="gpt-3.5-turbo"  


_EMBEDDER: Optional[Embedder] = None

_CLUSTER_PRINCIPLES_CACHE: Dict[int, List[Dict[str, Any]]] = {}
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



def extract_when_clause(text: str) -> str:
    """
    从 principle_text 中提取用于检索的 WHEN 子句：
    - 原始格式一般为: "When ..., you should ..." 或 "When ..., avoid ..."
    - 我们只取第一个逗号之前的部分作为"用户状态描述"。
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
    _CLUSTER_PRINCIPLES_EMB_CACHE[cache_key] = []
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

    cache_key = f"{cluster_config}_{cluster_id}"
    
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
        item["similarity"] = sim
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


from src.utils.load import load_dev_dataset
from src.utils.load import load_personas

class IncrementalSaver:
    """
    跨进程安全的增量保存器，支持实时保存结果到 JSON 文件
    使用文件锁保证多个进程同时写入时的安全性
    """
    def __init__(self, output_file: str):
        self.output_file = output_file
        self.lock_file = output_file + ".lock"
        self.completed_indices = set()
        
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if os.path.exists(output_file):
            try:
                with FileLock(self.lock_file, timeout=10):
                    with open(output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if isinstance(existing_results, list):
                            for result in existing_results:
                                if "persona_index" in result:
                                    self.completed_indices.add(result["persona_index"])
                            print(f"[Resume] Loaded {len(existing_results)} existing results, {len(self.completed_indices)} personas completed")
            except Exception as e:
                print(f"[Warning] Failed to load existing file: {e}, starting fresh")
                self.completed_indices = set()
    
    def save_result(self, result: Dict[str, Any]):
        """
        跨进程安全地保存单个结果（使用文件锁）
        """
        with FileLock(self.lock_file, timeout=30):
            existing_results = []
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if not isinstance(existing_results, list):
                            existing_results = []
                except Exception as e:
                    print(f"[Warning] Failed to read existing results: {e}, starting fresh")
                    existing_results = []
            
            persona_index = result.get("persona_index")
            if persona_index is not None:
                already_exists = any(
                    r.get("persona_index") == persona_index 
                    for r in existing_results
                )
                if already_exists:
                    existing_results = [
                        r for r in existing_results 
                        if r.get("persona_index") != persona_index
                    ]
                    existing_results.append(result)
                else:
                    existing_results.append(result)
                    self.completed_indices.add(persona_index)
            else:
                existing_results.append(result)
            
            temp_file = self.output_file + ".tmp"
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(existing_results, f, ensure_ascii=False, indent=2)
                os.replace(temp_file, self.output_file)
            except Exception as e:
                print(f"[Error] Failed to save result: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                raise
    
    def is_completed(self, index: int) -> bool:
        """
        检查指定索引（persona_index）的任务是否已完成
        需要读取文件来检查（因为不同进程的内存状态不同）
        """
        with FileLock(self.lock_file, timeout=10):
            if os.path.exists(self.output_file):
                try:
                    with open(self.output_file, "r", encoding="utf-8") as f:
                        existing_results = json.load(f)
                        if isinstance(existing_results, list):
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


PERSUADER_PROMPT_TYPE = "Ours_1"
CRITIC_PROMPT_TYPE = "LDPP"
PERSUADEE_PROMPT_TYPE = "personas"

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
) -> List[Dict[str, Any]]:
    """Run simulation using the initial dialogue (retrieve strategy chain + principles for guidance)."""
    if persuader_prompt_type is None:
        persuader_prompt_type = PERSUADER_PROMPT_TYPE
    if persuadee_prompt_type is None:
        persuadee_prompt_type = PERSUADEE_PROMPT_TYPE
    if critic_prompt_type is None:
        critic_prompt_type = CRITIC_PROMPT_TYPE

    import time

    dialog_history: List[Dict[str, Any]] = list(initial_dialog)

    current_turn_id = max((turn.get("turn_id", 0) for turn in dialog_history), default=0) + 1

    while True:
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

        strategy_hint_text = format_strategy_hint(retrieved_chains, depth=3)

        cluster_id = None
        if retrieved_chains:
            c0 = retrieved_chains[0]
            if isinstance(c0, dict):
                cluster_id = c0.get("cluster_id") or c0.get("cluster") or c0.get("cluster_idx")

        top_principles: List[Dict[str, Any]] = []
        query_text = get_last_persuadee_utterance(dialog_history)

        if isinstance(cluster_id, int):
            top_principles = get_top_principles_for_cluster(
                cluster_id=cluster_id,
                query_text=query_text,
                top_k=5,
                cluster_config=cluster_config,
            )

        time2 = time.time()

        # 3) breadth：principles → high_level_guidance
        high_level_guidance: Optional[str] = generate_high_level_guidance(
            principles=top_principles,
            last_user_utt=query_text,
            model=PERSUADER_GPT_MODEL,
            temperature=0.5,
            max_tokens=100
        )

        # 4) depth：strategy chain summary
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

        time3 = time.time()
        print(f"Guidance and summary generation time: {time3-time2:.2f}s")

        persuader_text = generate_persuader_utterance(
            dialog_history=dialog_history,
            strategy_chain_hint=used_strategy_hint_text,
            guidance_text=high_level_guidance,
            prompt_type=persuader_prompt_type,
            model=PERSUADER_GPT_MODEL
        )

        time4 = time.time()
        print(f"Persuader utterance generation time: {time4-time3:.2f}s")

        if verbose:
            print(f"[persuader]: {persuader_text}")
            print("--------------------------------")

        persuader_turn = {
            "turn_id": current_turn_id,
            "speaker": "Persuader",
            "text": persuader_text,
            "strategy_hint": retrieved_chains,
            "strategy_hint_text": used_strategy_hint_text,
            "principle_hint": top_principles,
            "high_level_guidance": high_level_guidance,
        }
        dialog_history.append(persuader_turn)

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
                print(f"Persuadee attitude={attitude}, ending dialog")
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
        sample = random.choice(dataset)
        base_index = sample.get("index", f"sample_{i}")
        dialog = sample.get("dialog", [])

        persona_idx = i  # 0..num_personas-1
        persona_description = persona_list[persona_idx]

        index = f"{base_index}_persona{persona_idx}"

        if len(dialog) >= 2:
            initial_turns = dialog[:2]
        elif len(dialog) >= 1:
            initial_turns = dialog[:1]
        else:
            return None

        simulated_dialog = run_simulation_with_initial(
            initial_dialog=initial_turns,
            max_turns=max_turns,
            persuader_prompt_type=persuader_prompt_type,
            critic_prompt_type=critic_prompt_type,
            persuadee_prompt_type=persuadee_prompt_type,
            persona_description=persona_description,
            cluster_config=cluster_config,
            verbose=verbose,
        )

        result = {
            "original_index": index,
            "dev_index": base_index,
            "persona_used": persona_description,
            "persona_index": persona_idx,
            "initial_turns": len(initial_turns),
            "simulated_dialog": simulated_dialog,
            "total_turns": len(simulated_dialog) / 2,
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
    print("Loading dataset...")
    dataset = load_dev_dataset()
    personas = load_personas()
    persona_list = list(personas.values())

    if not dataset:
        print("Failed to load dataset")
        return []

    if not persona_list:
        print("No personas loaded, exiting")
        return []

    num_personas = len(persona_list)  # e.g., 40
    if max_samples is None:
        total_runs = num_personas
    else:
        total_runs = min(max_samples, num_personas)

    if start_index is not None:
        actual_start = start_index
    else:
        actual_start = 0
    
    if end_index is not None:
        actual_end = min(end_index, total_runs - 1)
    else:
        actual_end = total_runs - 1

    actual_total = actual_end - actual_start + 1

    saver = None
    if enable_incremental_save and output_file:
        saver = IncrementalSaver(output_file)
        completed_count = saver.get_completed_count()
        if completed_count > 0:
            print(f"[Resume] Found {completed_count} completed tasks, skipping")
    else:
        print("[Info] Incremental save disabled; results will be saved after all tasks complete")

    print(f"Starting batch evaluation, index range: {actual_start}..{actual_end}, total {actual_total} runs")
    print("Note: parallelism is managed by the shell script; this process handles its assigned slice serially")

    results: List[Dict[str, Any]] = []
    failed_count = 0
    skipped_count = 0

    task_indices = []
    for i in range(actual_start, actual_end + 1):
        if saver and saver.is_completed(i):
            skipped_count += 1
            continue
        task_indices.append(i)

    if len(task_indices) == 0:
        print(f"[Info] All tasks already completed ({skipped_count} skipped)")
        if saver:
            with FileLock(saver.lock_file, timeout=10):
                if os.path.exists(saver.output_file):
                    try:
                        with open(saver.output_file, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except Exception:
                        pass
        return []

    with tqdm(total=actual_total, desc="Progress", unit="dialog", initial=skipped_count) as pbar:
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
                )

                if result is not None:
                    results.append(result)
                    if saver:
                        saver.save_result(result)
                    total_success = saver.get_completed_count() if saver else len(results)
                    pbar.set_postfix({
                        "success": total_success,
                        "failed": failed_count,
                        "skipped": skipped_count
                    })
                else:
                    failed_count += 1
                    total_success = saver.get_completed_count() if saver else len(results)
                    pbar.set_postfix({
                        "success": total_success,
                        "failed": failed_count,
                        "skipped": skipped_count
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

    print(f"\nBatch evaluation done: {len(all_results)} succeeded, {failed_count} failed, {skipped_count} skipped")
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
        help="cluster config name, e.g. kmeans_k50, kmeans_k100, kmeans_k150",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="enable verbose output (recommended off for parallel runs)",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=None,
        help="start index for batch processing (None = start from 0)",
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=None,
        help="end index for batch processing (None = run to max_samples)",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="output file path (None = auto-generate when incremental save enabled)",
    )
    parser.add_argument(
        "--disable_incremental_save",
        action="store_true",
        help="disable incremental save (enabled by default)",
    )

    args = parser.parse_args()

    if args.output_file:
        output_file = args.output_file
    else:
        output_file = f"outputs/P4G/evaluate/history/{args.cluster_config}/results_{PERSUADER_PROMPT_TYPE}_{PERSUADEE_PROMPT_TYPE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    enable_incremental_save = not args.disable_incremental_save

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
    )

    if not enable_incremental_save:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to {output_file}")
    else:
        print(f"Results saved incrementally to {output_file}")
    
    print(f"Evaluated {len(results)} dialogs")

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
