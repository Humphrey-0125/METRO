#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Unified Top-K strategy chain builder for CB / ESC / P4G.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs):
        return x


def load_states(states_path: str) -> List[Dict[str, Any]]:
    with open(states_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("states_embeddings.json must be a JSON list.")
    return data


def load_id2cluster(labels_path: str) -> Dict[str, int]:
    id2c: Dict[str, int] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            clu = obj.get("cluster")
            if rid is None or clu is None:
                continue
            id2c[str(rid)] = int(clu)
    if not id2c:
        raise ValueError("No id->cluster mapping read from labels.jsonl.")
    return id2c


def _load_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    objs: List[Dict[str, Any]] = []
    if p.suffix.lower() == ".jsonl":
        with p.open("r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    obj = json.loads(ln)
                except json.JSONDecodeError:
                    continue
                if isinstance(obj, dict):
                    objs.append(obj)
    else:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            objs = [x for x in data if isinstance(x, dict)]
        elif isinstance(data, dict):
            objs = [data]
        else:
            raise ValueError("Dialog results must be JSON object/list or JSONL.")
    return objs


def load_dialog_meta(
    dialog_path: str,
    dataset: str,
    max_turns_fallback: int = 10,
) -> Dict[str, Dict[str, Any]]:
    dataset = (dataset or "").strip().lower()
    objs = _load_json_or_jsonl(dialog_path)
    meta: Dict[str, Dict[str, Any]] = {}

    if dataset == "cb":
        for obj in objs:
            did = obj.get("dialogue_id") or obj.get("id")
            if did is None:
                continue
            did = str(did)

            deal = bool(obj.get("deal", False))
            num_turns = obj.get("num_turns", max_turns_fallback)
            try:
                num_turns = int(num_turns) if num_turns is not None else max_turns_fallback
            except Exception:
                num_turns = max_turns_fallback
            num_turns = max(num_turns, 1)

            def _to_float(v: Any) -> Optional[float]:
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None

            price_f = _to_float(obj.get("price"))
            buyer_f = _to_float(obj.get("buyer_price"))
            seller_f = _to_float(obj.get("seller_price"))

            sl = 0.0
            if deal and price_f is not None and buyer_f is not None and seller_f is not None:
                if buyer_f != seller_f:
                    sl = (price_f - seller_f) / (buyer_f - seller_f)

            meta[did] = {
                "num_turns": num_turns,
                "success": deal,
                "sl": float(sl),
            }
    elif dataset in {"esc", "p4g"}:
        for obj in objs:
            did = obj.get("dialogue_id")
            if did is None:
                continue

            att = obj.get("final_attitude_prediction", "B")
            num_turns = obj.get("num_turns", 1)
            try:
                num_turns = int(num_turns) if num_turns is not None else 1
            except Exception:
                num_turns = 1
            num_turns = max(num_turns, 1)

            att = str(att) if att is not None else "B"
            meta[str(did)] = {
                "num_turns": num_turns,
                "attitude": att,
                "success": (att == "D"),
            }
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use cb / esc / p4g.")

    if not meta:
        raise ValueError("No dialog metadata parsed from dialogs file.")
    return meta


def parse_state_id(state_id: str) -> Tuple[str, int]:
    if not isinstance(state_id, str):
        return "unknown", -1
    parts = state_id.split(":")
    if len(parts) < 2:
        return parts[0], -1
    did = parts[0]
    try:
        tid = int(parts[1])
    except Exception:
        tid = -1
    return did, tid


def normalize_strategy_chain(chain: Any) -> List[str]:
    if not isinstance(chain, list):
        return []
    out: List[str] = []
    for step in chain:
        if isinstance(step, str):
            t = step.strip()
            if t:
                out.append(t)
        elif isinstance(step, (list, tuple)):
            toks = [str(x).strip() for x in step if isinstance(x, str) and str(x).strip()]
            if toks:
                out.append("[" + ", ".join(toks) + "]")
        else:
            t = str(step).strip()
            if t:
                out.append(t)
    return out


def make_reward_fn(dialog_meta: Dict[str, Dict[str, Any]], dataset: str, mode: str):
    dataset = (dataset or "").strip().lower()
    mode = (mode or "").strip().lower()

    if dataset == "cb":
        if mode == "sl":
            def reward(did: str) -> Optional[float]:
                rec = dialog_meta.get(did)
                if not rec:
                    return None
                return float(rec.get("sl", 0.0) or 0.0)
            return reward

        if mode == "binary_deal":
            def reward(did: str) -> Optional[float]:
                rec = dialog_meta.get(did)
                if not rec:
                    return None
                return 1.0 if bool(rec.get("success", False)) else 0.0
            return reward

        raise ValueError("CB reward mode must be one of: sl, binary_deal.")

    if dataset in {"esc", "p4g"}:
        if mode == "attitude_score":
            score_map = {"A": -1.0, "B": -0.5, "C": 0.1, "D": 1.0}

            def reward(did: str) -> Optional[float]:
                rec = dialog_meta.get(did)
                if not rec:
                    return None
                att = rec.get("attitude")
                if att not in score_map:
                    return None
                return float(score_map[att])
            return reward

        if mode == "binary_d":
            def reward(did: str) -> Optional[float]:
                rec = dialog_meta.get(did)
                if not rec:
                    return None
                att = rec.get("attitude")
                if att not in {"A", "B", "C", "D"}:
                    return None
                return 1.0 if att == "D" else 0.0
            return reward

        raise ValueError("ESC/P4G reward mode must be one of: attitude_score, binary_d.")

    raise ValueError(f"Unknown dataset: {dataset}.")


def make_relative_penalty_fn(dialog_meta: Dict[str, Dict[str, Any]], lambda_len: float, alpha: float = 1.0):
    lam = float(lambda_len)
    a = float(alpha)

    def penalty(did: str, turn_id: int) -> float:
        if lam <= 0:
            return 0.0
        rec = dialog_meta.get(did, {})
        num_turns = int(rec.get("num_turns", 1) or 1)
        num_turns = max(num_turns, 1)
        tid = max(int(turn_id), -1)
        pos = (tid + 1) / num_turns
        pos = max(0.0, min(1.0, pos))
        return lam * (pos ** a)

    return penalty


def is_success(dialog_meta: Dict[str, Dict[str, Any]], did: str) -> Optional[bool]:
    rec = dialog_meta.get(did)
    if not rec:
        return None
    return bool(rec.get("success", False))


def _node_new(name: str) -> Dict[str, Any]:
    return {
        "name": name,
        "count": 0,
        "value_sum": 0.0,
        "success": 0,
        "children": {},
    }


def backprop_with_discount(
    root: Dict[str, Any],
    path: List[str],
    base_value: float,
    gamma: float,
    success_flag: bool,
):
    node = root
    node["count"] += 1
    node["value_sum"] += float(base_value)
    if success_flag:
        node["success"] += 1

    for d, tok in enumerate(path, start=1):
        ch = node["children"].get(tok)
        if ch is None:
            ch = _node_new(tok)
            node["children"][tok] = ch

        w = gamma ** d
        ch["count"] += 1
        ch["value_sum"] += float(base_value) * w
        if success_flag:
            ch["success"] += 1
        node = ch


def _prune(node: Dict[str, Any], min_count: int) -> Dict[str, Any]:
    if node.get("name") != "ROOT" and node.get("count", 0) < min_count:
        return {}
    new_children = {}
    for k, ch in node.get("children", {}).items():
        pruned = _prune(ch, min_count)
        if pruned:
            new_children[k] = pruned
    node["children"] = new_children
    return node


def _finalize(node: Dict[str, Any]) -> Dict[str, Any]:
    cnt = max(int(node.get("count", 0)), 1)
    node["avg_value"] = float(node.get("value_sum", 0.0)) / cnt
    node["success_rate"] = float(node.get("success", 0)) / cnt
    children = node.get("children", {})
    out_children = []
    for k in sorted(children.keys()):
        out_children.append(_finalize(children[k]))
    node["children"] = out_children
    return node


def wilson_lower_bound(success: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    phat = success / n
    denom = 1.0 + (z * z) / n
    center = phat + (z * z) / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)
    return max(0.0, (center - margin) / denom)


def _children_dict_from_finalized(node: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out = {}
    for ch in node.get("children", []) or []:
        name = ch.get("name")
        if name is not None:
            out[str(name)] = ch
    return out


def beam_search_topk(
    finalized_root: Dict[str, Any],
    top_k: int = 3,
    beam_width: int = 10,
    max_depth: int = 6,
    w_sr: float = 1.0,
    w_val: float = 0.2,
    w_cnt: float = 0.05,
    z: float = 1.96,
) -> List[Dict[str, Any]]:
    root = dict(finalized_root)
    root_children = _children_dict_from_finalized(root)

    beams: List[Tuple[List[str], float, Dict[str, Any]]] = []
    for tok, ch in root_children.items():
        cnt = int(ch.get("count", 0) or 0)
        succ = int(ch.get("success", 0) or 0)
        sr_lb = wilson_lower_bound(succ, cnt, z=z)
        av = float(ch.get("avg_value", 0.0) or 0.0)
        score = w_sr * sr_lb + w_val * av + w_cnt * math.log1p(cnt)
        beams.append(([tok], score, ch))

    beams.sort(key=lambda x: x[1], reverse=True)
    beams = beams[:beam_width]

    completed: List[Tuple[List[str], float, Dict[str, Any]]] = []
    for _depth in range(2, max_depth + 1):
        new_beams: List[Tuple[List[str], float, Dict[str, Any]]] = []
        for seq, sc, node in beams:
            ch_map = _children_dict_from_finalized(node)
            if not ch_map:
                completed.append((seq, sc, node))
                continue
            for tok, ch in ch_map.items():
                cnt = int(ch.get("count", 0) or 0)
                succ = int(ch.get("success", 0) or 0)
                sr_lb = wilson_lower_bound(succ, cnt, z=z)
                av = float(ch.get("avg_value", 0.0) or 0.0)
                node_score = w_sr * sr_lb + w_val * av + w_cnt * math.log1p(cnt)
                new_beams.append((seq + [tok], sc + node_score, ch))

        if not new_beams:
            break
        new_beams.sort(key=lambda x: x[1], reverse=True)
        beams = new_beams[:beam_width]

    completed.extend(beams)

    seen = set()
    uniq = []
    for seq, sc, node in sorted(completed, key=lambda x: x[1], reverse=True):
        key = " -> ".join(seq)
        if key in seen:
            continue
        seen.add(key)
        uniq.append((seq, sc, node))
        if len(uniq) >= top_k:
            break

    out = []
    for seq, sc, node in uniq:
        out.append(
            {
                "chain": seq,
                "score": float(sc),
                "end_node": {
                    "count": int(node.get("count", 0) or 0),
                    "success": int(node.get("success", 0) or 0),
                    "success_rate": float(node.get("success_rate", 0.0) or 0.0),
                    "avg_value": float(node.get("avg_value", 0.0) or 0.0),
                },
            }
        )
    return out


def _default_reward_mode(dataset: str) -> str:
    return "sl" if dataset == "cb" else "attitude_score"


def build_arg_parser(default_dataset: Optional[str] = None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    if default_dataset:
        ap.add_argument(
            "--dataset",
            type=str,
            default=default_dataset,
            choices=["cb", "esc", "p4g"],
            help=f"dataset type (default: {default_dataset})",
        )
    else:
        ap.add_argument(
            "--dataset",
            type=str,
            required=True,
            choices=["cb", "esc", "p4g"],
            help="dataset type",
        )

    ap.add_argument("--states", required=True, help="states_embeddings.json (list with id, strategy_chain)")
    ap.add_argument("--labels", required=True, help="labels.jsonl (id->cluster)")
    ap.add_argument("--dialogs", required=True, help="dialog results (.json/.jsonl)")
    ap.add_argument("--outdir", required=True, help="output dir")

    ap.add_argument(
        "--reward-mode",
        type=str,
        default=None,
        help="CB: sl/binary_deal; ESC/P4G: attitude_score/binary_d",
    )
    ap.add_argument("--lambda-len", type=float, default=0.0, help="relative length penalty coeff")
    ap.add_argument("--alpha", type=float, default=1.0, help="penalty exponent")
    ap.add_argument("--gamma", type=float, default=0.9, help="discount gamma")
    ap.add_argument("--min-count", type=int, default=1, help="prune threshold")
    ap.add_argument("--dedup-dialog-per-cluster", action="store_true", help="dedup same dialogue per cluster")

    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--beam", type=int, default=10)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--w-sr", type=float, default=1.0)
    ap.add_argument("--w-val", type=float, default=0.2)
    ap.add_argument("--w-cnt", type=float, default=0.05)
    ap.add_argument("--z", type=float, default=1.96)

    ap.add_argument(
        "--max-turns-fallback",
        type=int,
        default=10,
        help="fallback num_turns for CB when missing",
    )
    return ap


def main(default_dataset: Optional[str] = None):
    args = build_arg_parser(default_dataset=default_dataset).parse_args()
    dataset = (args.dataset or "").strip().lower()

    states = load_states(args.states)
    id2cluster = load_id2cluster(args.labels)
    dialog_meta = load_dialog_meta(
        args.dialogs,
        dataset=dataset,
        max_turns_fallback=int(args.max_turns_fallback),
    )

    reward_mode = args.reward_mode or _default_reward_mode(dataset)
    reward_fn = make_reward_fn(dialog_meta, dataset=dataset, mode=reward_mode)
    pen_fn = make_relative_penalty_fn(dialog_meta, lambda_len=args.lambda_len, alpha=args.alpha)

    cluster2items: Dict[int, List[Tuple[str, int, List[str]]]] = defaultdict(list)
    for obj in states:
        rid = obj.get("id")
        chain = obj.get("strategy_chain")
        if not rid or chain is None:
            continue
        clu = id2cluster.get(str(rid))
        if clu is None:
            continue
        did, tid = parse_state_id(str(rid))
        clean = normalize_strategy_chain(chain)
        if not clean:
            continue
        cluster2items[int(clu)].append((did, tid, clean))

    out_base = Path(args.outdir)
    out_base.mkdir(parents=True, exist_ok=True)
    trees_dir = out_base / "trees"
    trees_dir.mkdir(parents=True, exist_ok=True)
    topk_dir = out_base / "topk"
    topk_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {dataset}")
    print(f"Building trees into: {trees_dir}")
    print(f"Exporting top-k chains into: {topk_dir}")

    for clu, items in tqdm(sorted(cluster2items.items()), desc="Build+TopK"):
        root = _node_new("ROOT")
        seen_dialogs = set()

        for did, tid, path in items:
            base_r = reward_fn(did)
            succ_flag = is_success(dialog_meta, did)
            if base_r is None or succ_flag is None:
                continue

            if args.dedup_dialog_per_cluster:
                key = (clu, did)
                if key in seen_dialogs:
                    continue
                seen_dialogs.add(key)

            pen = pen_fn(did, tid)
            val = float(base_r) - float(pen)

            backprop_with_discount(
                root=root,
                path=path,
                base_value=val,
                gamma=float(args.gamma),
                success_flag=bool(succ_flag),
            )

        if args.min_count > 1:
            root = _prune(root, int(args.min_count)) or root

        finalized = _finalize(root)
        top_chains = beam_search_topk(
            finalized_root=finalized,
            top_k=int(args.topk),
            beam_width=int(args.beam),
            max_depth=int(args.max_depth),
            w_sr=float(args.w_sr),
            w_val=float(args.w_val),
            w_cnt=float(args.w_cnt),
            z=float(args.z),
        )

        tree_payload = {
            "dataset": dataset,
            "cluster": int(clu),
            "n_items_raw": int(len(items)),
            "config": {
                "reward_mode": reward_mode,
                "lambda_len": float(args.lambda_len),
                "alpha": float(args.alpha),
                "gamma": float(args.gamma),
                "min_count": int(args.min_count),
                "dedup_dialog_per_cluster": bool(args.dedup_dialog_per_cluster),
            },
            "tree": finalized,
        }
        with (trees_dir / f"cluster_{int(clu)}.json").open("w", encoding="utf-8") as f:
            json.dump(tree_payload, f, ensure_ascii=False, indent=2)

        topk_payload = {
            "dataset": dataset,
            "cluster": int(clu),
            "topk": int(args.topk),
            "beam": int(args.beam),
            "max_depth": int(args.max_depth),
            "weights": {"w_sr": args.w_sr, "w_val": args.w_val, "w_cnt": args.w_cnt, "z": args.z},
            "top_chains": top_chains,
        }
        with (topk_dir / f"cluster_{int(clu)}_top{int(args.topk)}.json").open("w", encoding="utf-8") as f:
            json.dump(topk_payload, f, ensure_ascii=False, indent=2)

    print("Done. Trees and Top-K chains exported.")


if __name__ == "__main__":
    main()
