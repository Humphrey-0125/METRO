import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm

try:
    import ijson
except ImportError:
    ijson = None

from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL

def build_turn_map(dialog: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    将 dialog 转为按 turn_id 聚合的字典：
    {
      0: {"Persuader": {...}, "Persuadee": {...}},
      1: {...},
      ...
    }
    """
    turn_map: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for entry in dialog:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("turn_id")
        if not isinstance(tid, int):
            continue
        speaker = entry.get("speaker") or entry.get("role") or "unknown"
        if tid not in turn_map:
            turn_map[tid] = {}
        turn_map[tid][speaker] = entry
    return turn_map

def concat_history_utterances_from_map(turn_map: Dict[int, Dict[str, Dict[str, Any]]],
                                       current_turn_id: int,
                                       max_history_turns: Optional[int] = None) -> str:
    """
    将所有 tid < current_turn_id 的 Persuader 与 Persuadee 的 text 按升序 tid 拼接为 history_text。
    可选限制最近 N 轮：max_history_turns (None 表示不限制)。
    输出空字符串表示没有历史发言可用。
    """
    parts: List[str] = []
    sorted_tids = sorted([tid for tid in turn_map.keys() if isinstance(tid, int) and tid < current_turn_id])
    if max_history_turns is not None and max_history_turns > 0:
        sorted_tids = sorted_tids[-max_history_turns:]
    for tid in sorted_tids:
        entry_dict = turn_map.get(tid, {})
        ordered_speakers = []
        if "Persuader" in entry_dict:
            ordered_speakers.append("Persuader")
        if "Persuadee" in entry_dict:
            ordered_speakers.append("Persuadee")
        for sp in entry_dict:
            if sp not in ordered_speakers:
                ordered_speakers.append(sp)
        for sp in ordered_speakers:
            entry = entry_dict.get(sp)
            if not entry:
                continue
            txt = entry.get("text") or entry.get("utterance") or ""
            if isinstance(txt, str) and txt.strip():
                parts.append(f"turn_{tid}|{sp}: {txt.strip()}")
    return " \n ".join(parts).strip()

def get_persuader_strategy_chain_from_map(turn_map: Dict[int, Dict[str, Dict[str, Any]]],
                                          start_turn_id: int) -> List[List[str]]:
    """
    从 start_turn_id 到最大 turn_id，按每轮只收集 Persuader 条目的 strategy（保持每轮为 list）
    例如：[[s1,s2], [], [s3]]
    """
    if not turn_map:
        return []
    max_tid = max(turn_map.keys())
    chain: List[List[str]] = []
    for tid in range(start_turn_id, max_tid + 1):
        entry_dict = turn_map.get(tid, {})
        p_entry = entry_dict.get("Persuader") or entry_dict.get("persuader")
        strategies_for_turn: List[str] = []
        if isinstance(p_entry, dict):
            ss = p_entry.get("strategy")
            if isinstance(ss, list):
                for s in ss:
                    if isinstance(s, str) and s.strip():
                        strategies_for_turn.append(s.strip())
        # 去重但保持顺序
        seen = set()
        cleaned = []
        for s in strategies_for_turn:
            if s not in seen:
                cleaned.append(s)
                seen.add(s)
        chain.append(cleaned)
    return chain

def get_pair_texts_from_map(turn_map: Dict[int, Dict[str, Dict[str, Any]]], turn_id: int) -> Dict[str, str]:
    """
    返回该轮的 persuader_text 与 persuadee_text（尽量从同一 turn_id 下找）。
    回退策略：
      - 如果一方缺失，尝试用前一轮的对应角色补齐；
      - 否则用空字符串。
    """
    persuader_text = ""
    persuadee_text = ""

    entry = turn_map.get(turn_id, {})
    p_entry = entry.get("Persuader") or entry.get("persuader")
    q_entry = entry.get("Persuadee") or entry.get("persuadee")

    if p_entry:
        persuader_text = p_entry.get("text") or p_entry.get("utterance") or ""
    if q_entry:
        persuadee_text = q_entry.get("text") or q_entry.get("utterance") or ""

    # fallback: try previous turn
    if not persuader_text:
        prev = turn_map.get(turn_id - 1, {})
        prev_p = prev.get("Persuader") or prev.get("persuader")
        if prev_p:
            persuader_text = prev_p.get("text") or prev_p.get("utterance") or ""
    if not persuadee_text:
        prev = turn_map.get(turn_id - 1, {})
        prev_q = prev.get("Persuadee") or prev.get("persuadee")
        if prev_q:
            persuadee_text = prev_q.get("text") or prev_q.get("utterance") or ""

    return {"persuader": persuader_text or "", "persuadee": persuadee_text or ""}

def stream_samples(path: Path):
    if ijson is not None:
        with path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                yield obj
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                yield obj
        else:
            raise ValueError("Input JSON must be an array.")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--api-url", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--history-max-turns", type=int, default=None,
                    help="若需限制历史轮数（只保留最近 N 轮），传 N；默认不限制")
    args = ap.parse_args()

    embedder = Embedder(
        model_name=args.model or EMBED_MODEL_NAME,
        api_key=args.api_key or EMBED_API_KEY,
        api_url=args.api_url or EMBED_API_URL,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fout:
        # 写入数组开始
        fout.write("[\n")
        wrote = 0
        first = True
        count = 0
        pbar = tqdm(total=args.max_items, desc="Embedding (utterance history)", ncols=100)

        for sample in stream_samples(Path(args.input)):
            index = sample.get("index", "unknown_index")
            dialog = sample.get("dialog", [])
            if not isinstance(dialog, list):
                continue

            turn_map = build_turn_map(dialog)
            # 只处理 turn_id >= 1 且该轮存在 Persuader 发言
            candidate_turn_ids = sorted([tid for tid in turn_map.keys()
                                        if isinstance(tid, int) and tid >= 1 and ("Persuader" in turn_map.get(tid, {}))])

            for tid in candidate_turn_ids:
                rec_id = f"{index}:{tid}"

                # 1) 构建 history_text：所有 tid < tid 的发言文本（Persuader+Persuadee 等）
                history_text = concat_history_utterances_from_map(turn_map, tid, max_history_turns=args.history_max_turns)
                if not history_text:
                    # fallback: 尝试使用上一轮（tid-1）双方发言拼接
                    fallback_parts = []
                    prev = turn_map.get(tid - 1, {})
                    if prev:
                        if "Persuader" in prev and prev["Persuader"].get("text"):
                            fallback_parts.append(f"turn_{tid-1}|Persuader: {prev['Persuader'].get('text')}")
                        if "Persuadee" in prev and prev["Persuadee"].get("text"):
                            fallback_parts.append(f"turn_{tid-1}|Persuadee: {prev['Persuadee'].get('text')}")
                    history_text = " \n ".join(fallback_parts).strip()

                # 2) 构造 strategy_chain：从当前轮开始，只收集 Persuader 条目的 strategy（二维列表）
                strategy_chain = get_persuader_strategy_chain_from_map(turn_map, tid)

                # 3) 获取本轮的 persuader / persuadee 文本
                pair_texts = get_pair_texts_from_map(turn_map, tid)
                current_persuader_text = pair_texts.get("persuader", "")
                current_persuadee_text = pair_texts.get("persuadee", "")

                # 4) 嵌入（使用 history_text）
                emb = embedder.encode(history_text)

                prev_id = f"{index}:{tid-1}"

                item = {
                    "id": rec_id,
                    "history_embedding": emb,
                    "strategy_chain": strategy_chain,
                    "prev_id": prev_id,
                    "persuader_text": current_persuader_text,
                    "persuadee_text": current_persuadee_text,
                    "history_text": history_text
                }

                # 以漂亮的多行缩进格式写入（streaming safe）
                item_json = json.dumps(item, ensure_ascii=False, indent=2)
                if not first:
                    fout.write(",\n")
                fout.write(item_json)
                first = False

                wrote += 1
                count += 1
                pbar.update(1)

                if args.sleep > 0:
                    time.sleep(args.sleep)
                if args.max_items and count >= args.max_items:
                    break
            if args.max_items and count >= args.max_items:
                break

        # 结束数组
        fout.write("\n]\n")
        pbar.close()

    print(f"✅ Wrote {wrote} items to {out_path}")

if __name__ == "__main__":
    main()

'''
# 基本用法（按回合合并所有 *_state）
python src/state/history_state_chain.py \
  --input data/P4G/merged_train_test.json \
  --output outputs/P4G/embedding/history/states_embeddings_train_test.json

'''