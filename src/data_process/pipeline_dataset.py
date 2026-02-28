#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _dataset_dir(dataset: str) -> str:
    m = {"p4g": "P4G", "cb": "CB", "esc": "ESC"}
    return m[dataset]


def _default_state_io(dataset: str):
    if dataset == "p4g":
        return (
            Path("data/P4G/merged_train_test.json"),
            Path("outputs/P4G/embedding/history/history_embeddings_train_test.json"),
        )
    if dataset == "cb":
        return (
            Path("data/CB/llm_expert.json"),
            Path("outputs/CB/embedding/history/history_embeddings_llm.json"),
        )
    return (
        Path("data/ESC/train.json"),
        Path("outputs/ESC/embedding/history/history_embeddings_train.json"),
    )


def _default_dialogs(dataset: str) -> Path:
    if dataset == "p4g":
        return Path("data/P4G/result.json")
    if dataset == "cb":
        return Path("data/CB/result_llm.json")
    return Path("data/ESC/result.json")


def _default_principles(dataset: str) -> Path:
    return Path(f"outputs/{_dataset_dir(dataset)}/principles/micro_principles.jsonl")


def _cluster_outdir(dataset: str, state_type: str, method: str, k: int) -> Path:
    base = Path(f"outputs/{_dataset_dir(dataset)}/cluster/{state_type}")
    if method == "kmeans":
        return base / f"kmeans_k{k}_llm"
    return base


def _run(cmd):
    print("[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def parse_steps(raw: str):
    s = raw.strip().lower()
    if s == "all":
        return ["state", "cluster", "mct", "split"]
    return [x.strip() for x in s.split(",") if x.strip()]


def main():
    ap = argparse.ArgumentParser(description="Run state->cluster->mct->split in one script.")
    ap.add_argument("--dataset", required=True, choices=["p4g", "cb", "esc"])
    ap.add_argument("--steps", default="all", help="all or comma list: state,cluster,mct,split")

    ap.add_argument("--state-type", default="history")

    ap.add_argument("--cluster-method", default="kmeans", choices=["kmeans", "optics", "hdbscan"])
    ap.add_argument("--k", type=int, default=100, help="used for kmeans output path")
    ap.add_argument("--normalize", action="store_true")

    ap.add_argument("--states", default=None, help="override states embedding path for mct")
    ap.add_argument("--labels", default=None, help="override labels.jsonl path for mct/split")
    ap.add_argument("--dialogs", default=None, help="override dialogs result path for mct")
    ap.add_argument("--outdir", default=None, help="override mct outdir")
    ap.add_argument("--principles", default=None, help="override principles path for split")

    ap.add_argument("--reward-mode", default=None)
    ap.add_argument("--lambda-len", type=float, default=0.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--min-count", type=int, default=1)
    ap.add_argument("--dedup-dialog-per-cluster", action="store_true")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--beam", type=int, default=20)
    ap.add_argument("--max-depth", type=int, default=9)
    ap.add_argument("--w-sr", type=float, default=1.0)
    ap.add_argument("--w-val", type=float, default=0.2)
    ap.add_argument("--w-cnt", type=float, default=0.05)
    ap.add_argument("--z", type=float, default=1.96)
    ap.add_argument("--max-turns-fallback", type=int, default=10)

    args = ap.parse_args()
    steps = parse_steps(args.steps)

    py = sys.executable
    src_dir = Path(__file__).resolve().parent

    _, default_states = _default_state_io(args.dataset)
    states_path = Path(args.states) if args.states else default_states

    # 2cluster.py expects this fixed input filename.
    cluster_expected_states = Path(f"outputs/{_dataset_dir(args.dataset)}/embedding/{args.state_type}/states_embeddings_train_test.json")

    cluster_out = _cluster_outdir(args.dataset, args.state_type, args.cluster_method, args.k)
    labels_path = Path(args.labels) if args.labels else (cluster_out / "labels.jsonl")

    dialogs_path = Path(args.dialogs) if args.dialogs else _default_dialogs(args.dataset)
    outdir_path = Path(args.outdir) if args.outdir else cluster_out
    principles_path = Path(args.principles) if args.principles else _default_principles(args.dataset)

    if "state" in steps:
        _run([py, str(src_dir / "1state.py"), "--task", args.dataset])

    if "cluster" in steps:
        if states_path.exists() and states_path.resolve() != cluster_expected_states.resolve():
            cluster_expected_states.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(states_path, cluster_expected_states)
            print(f"[INFO] Synced states to {cluster_expected_states}")

        task_upper = _dataset_dir(args.dataset)
        cmd = [
            py,
            str(src_dir / "2cluster.py"),
            "--task",
            task_upper,
            "--state_type",
            args.state_type,
            "--cluster-method",
            args.cluster_method,
        ]
        if args.normalize:
            cmd.append("--normalize")
        if args.cluster_method == "kmeans":
            cmd += ["--k", str(args.k)]

        _run(cmd)

    if "mct" in steps:
        cmd = [
            py,
            str(src_dir / "3MCT_topk.py"),
            "--dataset",
            args.dataset,
            "--states",
            str(states_path),
            "--labels",
            str(labels_path),
            "--dialogs",
            str(dialogs_path),
            "--outdir",
            str(outdir_path),
            "--lambda-len",
            str(args.lambda_len),
            "--alpha",
            str(args.alpha),
            "--gamma",
            str(args.gamma),
            "--min-count",
            str(args.min_count),
            "--topk",
            str(args.topk),
            "--beam",
            str(args.beam),
            "--max-depth",
            str(args.max_depth),
            "--w-sr",
            str(args.w_sr),
            "--w-val",
            str(args.w_val),
            "--w-cnt",
            str(args.w_cnt),
            "--z",
            str(args.z),
            "--max-turns-fallback",
            str(args.max_turns_fallback),
        ]
        if args.reward_mode:
            cmd += ["--reward-mode", args.reward_mode]
        if args.dedup_dialog_per_cluster:
            cmd.append("--dedup-dialog-per-cluster")

        _run(cmd)

    if "split" in steps:
        _run([
            py,
            str(src_dir / "4split_principles_by_cluster.py"),
            "--principles",
            str(principles_path),
            "--labels",
            str(labels_path),
        ])

    print("[DONE] pipeline finished")


if __name__ == "__main__":
    main()

"""
python src/pipeline_dataset.py --dataset p4g --steps all --k 100 --normalize
"""
