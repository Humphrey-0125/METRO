#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CB_DEFAULT_MAX_TURNS = 10
P4G_DEFAULT_MAPPING = {"A": -1.0, "B": -0.5, "C": 0.1, "D": 1.0}
ESC_DEFAULT_MAPPING = {"A": -1.0, "B": -0.5, "C": 0.5, "D": 1.0}
DEFAULT_THRESHOLD = 0.6
DEFAULT_CRITIC_KEY = "critic_attitudes"


def load_json_or_jsonl(path: str) -> Any:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        items = []
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    items.append(json.loads(ln))
                except json.JSONDecodeError:
                    continue
        return items

    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_evaluation_results(file_path: str) -> List[Dict[str, Any]]:
    data = load_json_or_jsonl(file_path)
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], list):
        return [x for x in data["results"] if isinstance(x, dict)]
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    raise ValueError(f"Unsupported JSON format in {file_path}: top-level type={type(data)}")


def _safe_int(x: Any) -> Optional[int]:
    try:
        if x is None:
            return None
        return int(float(x))
    except Exception:
        return None


# -------------------- CB --------------------
def _get_target_prices(r: Dict[str, Any]) -> Tuple[Any, Any]:
    p_buyer = r.get("buyer_price", None)
    p_seller = r.get("seller_price", None)
    if p_buyer is None or p_seller is None:
        meta = r.get("meta", {}) if isinstance(r.get("meta", {}), dict) else {}
        if p_buyer is None:
            p_buyer = meta.get("buyer_price", None)
        if p_seller is None:
            p_seller = meta.get("seller_price", None)
    return p_buyer, p_seller


def compute_cb_metrics(results: List[Dict[str, Any]], max_turns: int = CB_DEFAULT_MAX_TURNS) -> Dict[str, float]:
    if not results:
        return {"SR": 0.0, "AT": 0.0, "SL%": 0.0}

    total = len(results)
    success_dialogues = [r for r in results if bool(r.get("deal", False))]
    num_success = len(success_dialogues)

    sr = num_success / total
    at = (
        sum(float(r.get("num_turns", max_turns)) for r in success_dialogues) / num_success
        if num_success > 0
        else 0.0
    )

    sl_values: List[float] = []
    for r in results:
        if not bool(r.get("deal", False)):
            sl_values.append(0.0)
            continue

        p_deal = r.get("price", None)
        p_buyer, p_seller = _get_target_prices(r)
        if p_deal is None or p_buyer is None or p_seller is None:
            sl_values.append(0.0)
            continue

        try:
            p_deal_f = float(p_deal)
            p_buyer_f = float(p_buyer)
            p_seller_f = float(p_seller)
            if p_buyer_f == p_seller_f:
                sl_values.append(0.0)
                continue
            sl = (p_deal_f - p_seller_f) / (p_buyer_f - p_seller_f)
        except Exception:
            sl = 0.0
        sl_values.append(sl)

    sl_percent = sum(sl_values) / total
    return {"SR": round(sr, 4), "AT": round(at, 4), "SL%": round(sl_percent, 4)}


# -------------------- P4G / ESC --------------------
def parse_mapping(args: argparse.Namespace, dataset: str) -> Dict[str, float]:
    base = P4G_DEFAULT_MAPPING if dataset == "p4g" else ESC_DEFAULT_MAPPING
    mapping = dict(base)

    if args.mapping_file:
        data = load_json_or_jsonl(args.mapping_file)
        if not isinstance(data, dict):
            raise ValueError("--mapping_file must be a JSON object")
        mapping.update({str(k).strip().upper(): float(v) for k, v in data.items()})

    if args.mapping_json:
        data = json.loads(args.mapping_json)
        if not isinstance(data, dict):
            raise ValueError("--mapping_json must be a JSON object")
        mapping.update({str(k).strip().upper(): float(v) for k, v in data.items()})

    out = {}
    for k, v in mapping.items():
        ku = str(k).strip().upper()
        if ku in {"A", "B", "C", "D"}:
            out[ku] = float(v)
    return out


def _turn_score_from_critic_attitudes(turn: Dict[str, Any], mapping: Dict[str, float], critic_key: str) -> Optional[float]:
    cas = turn.get(critic_key)
    if not isinstance(cas, list) or len(cas) == 0:
        for alt_key in ["critic_attitudes", "sampled_attitudes"]:
            if alt_key == critic_key:
                continue
            alt_val = turn.get(alt_key)
            if isinstance(alt_val, list) and len(alt_val) > 0:
                cas = alt_val
                break

    if not isinstance(cas, list) or len(cas) == 0:
        return None

    vals: List[float] = []
    for a in cas:
        if not isinstance(a, str):
            continue
        au = a.strip().upper()
        if au in mapping:
            vals.append(mapping[au])

    if not vals:
        return None
    return sum(vals) / len(vals)


def _speaker_config(dataset: str) -> Tuple[str, str, bool]:
    if dataset == "p4g":
        return "persuadee", "simulated_dialog", False  # success if > threshold
    return "seeker", "simulated_dialog_or_dialog", True  # success if >= threshold


def _get_target_turns(dialogue: Dict[str, Any], dataset: str) -> List[Tuple[int, Dict[str, Any]]]:
    speaker, source, _ = _speaker_config(dataset)

    if source == "simulated_dialog":
        sim = dialogue.get("simulated_dialog", [])
    else:
        sim = dialogue.get("simulated_dialog")
        if sim is None:
            sim = dialogue.get("dialog", [])

    if not isinstance(sim, list):
        return []

    out = []
    for t in sim:
        if not isinstance(t, dict):
            continue
        sp = str(t.get("speaker", "")).strip().lower()
        if sp != speaker:
            continue
        tid = _safe_int(t.get("turn_id"))
        if tid is None:
            continue
        out.append((tid, t))

    out.sort(key=lambda x: x[0])
    return out


def _is_success(score: float, threshold: float, inclusive: bool) -> bool:
    return score >= threshold if inclusive else score > threshold


def get_final_score(dialogue: Dict[str, Any], dataset: str, mapping: Dict[str, float], critic_key: str) -> Optional[float]:
    turns = _get_target_turns(dialogue, dataset)
    if not turns:
        return None
    _, last_turn = turns[-1]
    return _turn_score_from_critic_attitudes(last_turn, mapping, critic_key)


def get_first_success_len(
    dialogue: Dict[str, Any],
    dataset: str,
    threshold: float,
    mapping: Dict[str, float],
    critic_key: str,
) -> Optional[int]:
    _, _, inclusive = _speaker_config(dataset)
    turns = _get_target_turns(dialogue, dataset)
    if not turns:
        return None

    for tid, turn in turns:
        s = _turn_score_from_critic_attitudes(turn, mapping, critic_key)
        if s is None:
            continue
        if _is_success(float(s), threshold, inclusive):
            return tid + 1
    return None


def compute_ssr_final(
    dialogues: List[Dict[str, Any]],
    dataset: str,
    mapping: Dict[str, float],
    critic_key: str,
    strict_valid: bool = False,
) -> Tuple[float, int, int]:
    scores: List[float] = []
    n_missing = 0

    for d in dialogues:
        s = get_final_score(d, dataset, mapping, critic_key)
        if s is None:
            n_missing += 1
            if not strict_valid:
                scores.append(0.0)
            continue
        scores.append(float(s))

    n_used = len(scores)
    ssr = (sum(scores) / n_used) if n_used else 0.0
    return ssr, n_used, n_missing


def compute_attitude_metrics(
    dialogues: List[Dict[str, Any]],
    dataset: str,
    threshold: float,
    mapping: Dict[str, float],
    critic_key: str,
    strict_valid: bool = False,
) -> Dict[str, Any]:
    n_total = len(dialogues)
    effective_total = n_total
    n_success = 0

    sum_success_len = 0.0
    n_no_target = 0
    n_missing_all_scores = 0

    for d in dialogues:
        turns = _get_target_turns(d, dataset)
        if not turns:
            n_no_target += 1
            continue

        found_any_score = False
        first_success_len: Optional[int] = None
        _, _, inclusive = _speaker_config(dataset)

        for tid, turn in turns:
            s = _turn_score_from_critic_attitudes(turn, mapping, critic_key)
            if s is None:
                continue
            found_any_score = True
            if _is_success(float(s), threshold, inclusive):
                first_success_len = tid + 1
                break

        if not found_any_score:
            n_missing_all_scores += 1
            if strict_valid:
                effective_total -= 1
            continue

        if first_success_len is not None:
            n_success += 1
            sum_success_len += float(first_success_len)

    sr = (n_success / effective_total) if effective_total else 0.0
    at = (sum_success_len / n_success) if n_success else 0.0
    ssr, ssr_n_used, ssr_missing = compute_ssr_final(
        dialogues, dataset, mapping, critic_key, strict_valid=strict_valid
    )

    return {
        "total_dialogues": n_total,
        "effective_total_dialogues": effective_total,
        "success_dialogues": n_success,
        "threshold": threshold,
        "mapping": mapping,
        "critic_key": critic_key,
        "sr": sr,
        "at": at,
        "sum_at": sum_success_len,
        "ssr": ssr,
        "ssr_n_used": ssr_n_used,
        "ssr_missing_final": ssr_missing,
        "no_target_speaker_dialogues": n_no_target,
        "missing_all_scores_dialogues": n_missing_all_scores,
    }


def evaluate_file(args: argparse.Namespace, path: str) -> Dict[str, Any]:
    results = load_evaluation_results(path)
    if args.dataset == "cb":
        m = compute_cb_metrics(results, max_turns=int(args.max_turns))
    else:
        mapping = parse_mapping(args, args.dataset)
        m = compute_attitude_metrics(
            dialogues=results,
            dataset=args.dataset,
            threshold=float(args.threshold),
            mapping=mapping,
            critic_key=str(args.critic_key),
            strict_valid=bool(args.strict_valid),
        )
    m["file_path"] = path
    return m


def list_input_files(input_path: str) -> List[str]:
    p = Path(input_path)
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = sorted([str(x) for x in p.glob("*.json")]) + sorted([str(x) for x in p.glob("*.jsonl")])
        return files
    raise ValueError(f"Path not found: {input_path}")


def main():
    ap = argparse.ArgumentParser(description="Unified metrics for CB / P4G / ESC")
    ap.add_argument("input_path", type=str, help="json/jsonl file or a directory")
    ap.add_argument("--dataset", required=True, choices=["cb", "p4g", "esc"])

    ap.add_argument("--max_turns", type=int, default=CB_DEFAULT_MAX_TURNS, help="CB only")

    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="P4G/ESC only")
    ap.add_argument("--mapping_json", type=str, default=None, help='P4G/ESC only, e.g. {"A":-1,"B":-0.5,"C":0.1,"D":1}')
    ap.add_argument("--mapping_file", type=str, default=None, help="P4G/ESC only")
    ap.add_argument("--critic_key", type=str, default=DEFAULT_CRITIC_KEY, help="P4G/ESC only")
    ap.add_argument("--strict_valid", action="store_true", help="P4G/ESC only")

    ap.add_argument("--detailed", action="store_true", help="directory mode: print per-file metrics")
    ap.add_argument("--save_json", type=str, default=None, help="save metrics json")
    args = ap.parse_args()

    files = list_input_files(args.input_path)
    if not files:
        raise ValueError("No .json/.jsonl files found.")

    all_results = [evaluate_file(args, f) for f in files]

    if len(all_results) == 1:
        out: Any = all_results[0]
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        if args.detailed:
            for r in all_results:
                print("=" * 80)
                print(r["file_path"])
                print(json.dumps(r, ensure_ascii=False, indent=2))

        if args.dataset == "cb":
            summary = {
                "dataset": "cb",
                "n_files": len(all_results),
                "avg_SR": round(sum(r.get("SR", 0.0) for r in all_results) / len(all_results), 4),
                "avg_AT": round(sum(r.get("AT", 0.0) for r in all_results) / len(all_results), 4),
                "avg_SL%": round(sum(r.get("SL%", 0.0) for r in all_results) / len(all_results), 4),
                "files": all_results,
            }
        else:
            summary = {
                "dataset": args.dataset,
                "n_files": len(all_results),
                "avg_sr": round(sum(r.get("sr", 0.0) for r in all_results) / len(all_results), 4),
                "avg_ssr": round(sum(r.get("ssr", 0.0) for r in all_results) / len(all_results), 4),
                "avg_at": round(sum(r.get("at", 0.0) for r in all_results) / len(all_results), 4),
                "files": all_results,
            }
        out = summary
        print(json.dumps(out, ensure_ascii=False, indent=2))

    if args.save_json:
        out_path = Path(args.save_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()

"""
python src/evaluate/metrics.py output/CB/result_llm.json --dataset cb --max_turns 10
"""