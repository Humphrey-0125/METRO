import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from tqdm import tqdm

try:
    import ijson
except ImportError:
    ijson = None

from src.utils.embed import Embedder, EMBED_MODEL_NAME, EMBED_API_KEY, EMBED_API_URL


# =============================
# Helpers
# =============================
def norm_speaker(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    return "unknown"


def safe_text(entry: Dict[str, Any]) -> str:
    # P4G/CB: text/utterance
    txt = entry.get("text") or entry.get("utterance")
    # ESC: content
    if txt is None:
        txt = entry.get("content")
    return txt.strip() if isinstance(txt, str) else ""


def normalize_strategy(v: Any) -> List[str]:
    """
    Normalize strategy to list[str], non-empty, dedup keep order.
    - P4G Persuader: list[str] or str
    - CB buyer: list[str] or str
    - ESC supporter: annotation.strategy (list[str] or str)
    """
    tmp: List[str] = []
    if isinstance(v, list):
        for s in v:
            if isinstance(s, str) and s.strip():
                tmp.append(s.strip())
    elif isinstance(v, str) and v.strip():
        tmp.append(v.strip())

    seen = set()
    out: List[str] = []
    for s in tmp:
        if s not in seen:
            out.append(s)
            seen.add(s)
    return out


def stream_samples(path: Path) -> Iterator[Dict[str, Any]]:
    """
    Expect input to be a JSON array (list of dicts).
    """
    if ijson is not None:
        with path.open("rb") as f:
            for obj in ijson.items(f, "item"):
                if isinstance(obj, dict):
                    yield obj
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            for obj in data:
                if isinstance(obj, dict):
                    yield obj
        else:
            raise ValueError("Input JSON must be an array.")


# =============================
# Turn utilities (turn-based)
# =============================
def build_turn_map(dialog: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    Standard format:
    entry has "turn_id"(int) and "speaker".
    turn_map: { tid: {speaker: entry} }
    """
    turn_map: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for entry in dialog:
        if not isinstance(entry, dict):
            continue
        tid = entry.get("turn_id")
        if not isinstance(tid, int):
            continue
        speaker = norm_speaker(entry.get("speaker") or entry.get("role") or "unknown").lower()
        if tid not in turn_map:
            turn_map[tid] = {}
        turn_map[tid][speaker] = entry
    return turn_map


def build_turn_map_esc(dialog: List[Dict[str, Any]]) -> Dict[int, Dict[str, Dict[str, Any]]]:
    """
    ESC train format:
    dialog is a flat list of turns:
      {"speaker":"supporter"/"seeker", "content":"...", "annotation":{"strategy":...}}
    No turn_id -> we group by idx//2 as turn_id.
    turn_map: { tid: {speaker: entry_with_turn_id} }
    """
    turn_map: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for idx, entry in enumerate(dialog):
        if not isinstance(entry, dict):
            continue
        tid = idx // 2  # supporter+seeker as one turn
        speaker = norm_speaker(entry.get("speaker") or "unknown").lower()

        # copy & attach turn_id for downstream compatibility
        e = dict(entry)
        e["turn_id"] = tid

        if tid not in turn_map:
            turn_map[tid] = {}
        turn_map[tid][speaker] = e

    return turn_map


def concat_history_utterances_from_map(
    turn_map: Dict[int, Dict[str, Dict[str, Any]]],
    current_turn_id: int,
    agent_role: str,
    other_role: str,
    max_history_turns: Optional[int] = None,
) -> str:
    """
    EXACTLY like your original P4G method:
    - history = all tid < current_turn_id
    - per turn speaker order: agent -> other -> rest
    """
    parts: List[str] = []
    sorted_tids = sorted(
        tid for tid in turn_map.keys()
        if isinstance(tid, int) and tid < current_turn_id
    )
    if max_history_turns is not None and max_history_turns > 0:
        sorted_tids = sorted_tids[-max_history_turns:]

    for tid in sorted_tids:
        entry_dict = turn_map.get(tid, {})
        ordered_speakers: List[str] = []
        if agent_role in entry_dict:
            ordered_speakers.append(agent_role)
        if other_role in entry_dict:
            ordered_speakers.append(other_role)
        for sp in entry_dict:
            if sp not in ordered_speakers:
                ordered_speakers.append(sp)

        for sp in ordered_speakers:
            entry = entry_dict.get(sp)
            if not entry:
                continue
            txt = safe_text(entry)
            if txt:
                parts.append(f"turn_{tid}|{sp}: {txt}")

    return " \n ".join(parts).strip()


def get_agent_strategy_chain_from_map(
    turn_map: Dict[int, Dict[str, Dict[str, Any]]],
    start_turn_id: int,
    agent_role: str,
    task: str,
) -> List[List[str]]:
    """
    From start_turn_id to max turn_id:
    collect agent_role's strategy per turn as list[str].

    task:
      - p4g/cb: entry.get("strategy")
      - esc: entry.get("annotation", {}).get("strategy")
    """
    if not turn_map:
        return []
    max_tid = max(tid for tid in turn_map.keys() if isinstance(tid, int))
    chain: List[List[str]] = []

    for tid in range(start_turn_id, max_tid + 1):
        entry_dict = turn_map.get(tid, {})
        a_entry = entry_dict.get(agent_role)
        strategies: List[str] = []

        if isinstance(a_entry, dict):
            if task == "esc":
                ann = a_entry.get("annotation") or {}
                if isinstance(ann, dict):
                    strategies = normalize_strategy(ann.get("strategy"))
                else:
                    strategies = []
            else:
                strategies = normalize_strategy(a_entry.get("strategy"))

        chain.append(strategies)

    return chain


def get_pair_texts_from_map(
    turn_map: Dict[int, Dict[str, Dict[str, Any]]],
    turn_id: int,
    agent_role: str,
    other_role: str,
) -> Dict[str, str]:
    """
    Same as your original:
    - read current turn
    - fallback to previous turn if missing
    """
    agent_text = ""
    other_text = ""

    entry = turn_map.get(turn_id, {})
    if agent_role in entry:
        agent_text = safe_text(entry[agent_role])
    if other_role in entry:
        other_text = safe_text(entry[other_role])

    if not agent_text:
        prev = turn_map.get(turn_id - 1, {})
        if agent_role in prev:
            agent_text = safe_text(prev[agent_role])
    if not other_text:
        prev = turn_map.get(turn_id - 1, {})
        if other_role in prev:
            other_text = safe_text(prev[other_role])

    return {"agent": agent_text or "", "other": other_text or ""}


# =============================
# Main (task)
# =============================
def main():
    """
    CLI arg: --task {p4g, cb, esc}

    - P4G: agent=persuader, other=persuadee, input uses dialog with turn_id
    - CB : agent=buyer, other=seller, input uses dialog with turn_id
    - ESC: agent=supporter, other=seeker, input uses flat dialog without turn_id,
           strategy stored at annotation.strategy
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["p4g", "cb", "esc", "P4G", "CB", "ESC"])
    args = ap.parse_args()

    task = args.task.lower()

    # --------- Hard-coded config ---------
    if task == "p4g":
        INPUT_PATH = "data/P4G/merged_train_test.json"
        OUTPUT_PATH = "outputs/P4G/embedding/history/history_embeddings_train_test.json"
        AGENT_ROLE = "persuader"
        OTHER_ROLE = "persuadee"
        MIN_TURN_ID = 1
        HISTORY_MAX_TURNS = None

    elif task == "cb":
        INPUT_PATH = "data/CB/llm_expert.json"
        OUTPUT_PATH = "outputs/CB/embedding/history/history_embeddings_llm.json"
        AGENT_ROLE = "buyer"
        OTHER_ROLE = "seller"
        MIN_TURN_ID = 1
        HISTORY_MAX_TURNS = None

    else:  # esc
        INPUT_PATH = "data/ESC/train.json"  # 你按实际路径改
        OUTPUT_PATH = "outputs/ESC/embedding/history/history_embeddings_train.json"
        AGENT_ROLE = "supporter"
        OTHER_ROLE = "seeker"
        MIN_TURN_ID = 1
        HISTORY_MAX_TURNS = None

    MAX_ITEMS = None
    SLEEP = 0.0

    MODEL_NAME = EMBED_MODEL_NAME
    API_URL = EMBED_API_URL
    API_KEY = EMBED_API_KEY
    # ------------------------------------

    embedder = Embedder(
        model_name=MODEL_NAME,
        api_key=API_KEY,
        api_url=API_URL,
    )

    in_path = Path(INPUT_PATH)
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    wrote = 0
    count = 0

    with out_path.open("w", encoding="utf-8") as fout:
        fout.write("[\n")
        first = True
        pbar = tqdm(total=MAX_ITEMS, desc=f"Embedding history ({task.upper()})", ncols=100)

        for sample_idx, sample in enumerate(stream_samples(in_path)):
            # -------- sample_id --------
            sample_id = None
            for k in ["id", "original_index", "index"]:
                if k in sample:
                    v = sample.get(k)
                    if isinstance(v, str) and v.strip():
                        sample_id = v.strip()
                        break
                    if isinstance(v, (int, float)):
                        sample_id = str(v)
                        break
            if not sample_id:
                sample_id = f"sample_{sample_idx}"

            # -------- dialog --------
            dialog = sample.get("dialog", [])
            if not isinstance(dialog, list):
                continue

            # -------- build turn_map --------
            if task == "esc":
                turn_map = build_turn_map_esc(dialog)
            else:
                turn_map = build_turn_map(dialog)

            # Candidate turns: only turns where AGENT_ROLE exists.
            candidate_turn_ids = sorted(
                tid for tid in turn_map.keys()
                if isinstance(tid, int)
                and tid >= MIN_TURN_ID
                and (AGENT_ROLE in turn_map.get(tid, {}))
            )

            # ESC meta (optional, but very useful)
            esc_meta = {}
            if task == "esc":
                esc_meta = {
                    "emotion_type": sample.get("emotion_type", ""),
                    "problem_type": sample.get("problem_type", ""),
                    "situation": sample.get("situation", ""),
                }

            print("process sample_id: ", sample_id)

            for tid in candidate_turn_ids:
                rec_id = f"{sample_id}:{tid}"
                prev_id = f"{sample_id}:{tid-1}"

                # 1) history_text (same style)
                history_text = concat_history_utterances_from_map(
                    turn_map=turn_map,
                    current_turn_id=tid,
                    agent_role=AGENT_ROLE,
                    other_role=OTHER_ROLE,
                    max_history_turns=HISTORY_MAX_TURNS,
                )

                # fallback: previous turn agent/other
                if not history_text:
                    fallback_parts: List[str] = []
                    prev = turn_map.get(tid - 1, {})
                    if AGENT_ROLE in prev and safe_text(prev[AGENT_ROLE]):
                        fallback_parts.append(f"turn_{tid-1}|{AGENT_ROLE}: {safe_text(prev[AGENT_ROLE])}")
                    if OTHER_ROLE in prev and safe_text(prev[OTHER_ROLE]):
                        fallback_parts.append(f"turn_{tid-1}|{OTHER_ROLE}: {safe_text(prev[OTHER_ROLE])}")
                    history_text = " \n ".join(fallback_parts).strip()

                # 2) strategy_chain: ONLY agent role
                strategy_chain = get_agent_strategy_chain_from_map(
                    turn_map=turn_map,
                    start_turn_id=tid,
                    agent_role=AGENT_ROLE,
                    task=task,
                )

                # 3) pair texts (agent/other)
                pair_texts = get_pair_texts_from_map(
                    turn_map=turn_map,
                    turn_id=tid,
                    agent_role=AGENT_ROLE,
                    other_role=OTHER_ROLE,
                )
                agent_text = pair_texts.get("agent", "")
                other_text = pair_texts.get("other", "")

                # 4) embedding
                print(f"Embedding history for {rec_id} ...")
                print("history_text: ", history_text)
                print("---------------------------------")
                emb = embedder.encode(history_text)

                item = {
                    "id": rec_id,
                    "prev_id": prev_id,
                    "turn_id": tid,
                    "agent_role": AGENT_ROLE,
                    "other_role": OTHER_ROLE,
                    "history_embedding": emb,
                    "strategy_chain": strategy_chain,  # ONLY agent chain
                    "history_text": history_text,
                }

                if task == "p4g":
                    item.update({"persuader_text": agent_text, "persuadee_text": other_text})
                elif task == "cb":
                    item.update({"buyer_text": agent_text, "seller_text": other_text})
                else:  # esc
                    item.update({
                        "supporter_text": agent_text,
                        "seeker_text": other_text,
                        "meta": esc_meta,
                    })

                item_json = json.dumps(item, ensure_ascii=False, indent=2)
                if not first:
                    fout.write(",\n")
                fout.write(item_json)
                first = False

                wrote += 1
                count += 1
                pbar.update(1)

                if SLEEP > 0:
                    time.sleep(SLEEP)
                if MAX_ITEMS and count >= MAX_ITEMS:
                    break

            if MAX_ITEMS and count >= MAX_ITEMS:
                break

        fout.write("\n]\n")
        pbar.close()

    print(f"✅ Wrote {wrote} items to {out_path}")


if __name__ == "__main__":
    main()

"""
Usage:
  python src/1state.py --task p4g
  python src/1state.py --task cb
  python src/1state.py --task esc
"""