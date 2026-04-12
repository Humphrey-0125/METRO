#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm

from src.utils.cluster import (
    load_ids_and_embeddings_auto,
    l2_normalize,
    kmeans_fit_predict,
    choose_k_by_silhouette,
    export_clusters,
    hdbscan_fit_predict,
    hdbscan_compute_centroids,
    optics_fit_predict,
    optics_compute_centroids,
)

def parse_klist(s: str):
    s = s.strip()
    if "," in s:
        return sorted({int(x) for x in s.split(",") if x.strip()})
    if "-" in s:
        a, b = s.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(s)] if s else None


def main():
    import time
    time1 = time.time()
    ap = argparse.ArgumentParser()

    ap.add_argument("--task", required=True, choices=["P4G", "CB", "ESC"], help="dataset task: p4g / cb / esc")

    ap.add_argument("--state_type", required=True, help="state type, used to build input/output paths")
    ap.add_argument("--normalize", action="store_true", help="apply L2 normalization to embeddings (for cosine similarity)")

    ap.add_argument(
        "--cluster-method",
        type=str,
        choices=["kmeans", "optics", "hdbscan"],
        default="kmeans",
        help="clustering method: kmeans / optics / hdbscan",
    )

    ap.add_argument("--k", type=int, default=None, help="number of clusters; if not set, grid search is used (kmeans only)")
    ap.add_argument("--k_grid", type=str, default=None, help="k grid range (e.g. '5-50' or '8,12,16'); default 5-50 step 5 (kmeans only)")
    ap.add_argument("--minibatch", action="store_true", help="use MiniBatchKMeans (more memory efficient, kmeans only)")
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=2048, help="MiniBatchKMeans batch size")

    ap.add_argument(
        "--use-gpu",
        action="store_true",
        help="use GPU KMeans (requires cuML & CuPy, kmeans only)",
    )

    ap.add_argument("--min-samples", type=int, default=5, help="OPTICS min_samples (minimum neighborhood size)")
    ap.add_argument("--xi", type=float, default=0.05, help="OPTICS xi parameter for cluster extraction")
    ap.add_argument("--min-cluster-size", type=int, default=5, help="OPTICS min_cluster_size (absolute minimum cluster size)")

    ap.add_argument("--hdb-min-cluster-size", type=int, default=15)
    ap.add_argument("--hdb-min-samples", type=int, default=None)
    ap.add_argument("--hdb-metric", type=str, default="euclidean", choices=["euclidean","cosine"])
    ap.add_argument("--hdb-select", type=str, default="eom", choices=["eom","leaf"])
    ap.add_argument("--hdb-allow-single", action="store_true")

    args = ap.parse_args()

    task = args.task.lower()
    if task == "p4g":
        task_dir = "P4G"
    elif task == "cb":
        task_dir = "CB"
    else:
        task_dir = "ESC"

    input_path = f"outputs/{task_dir}/embedding/{args.state_type}/states_embeddings_train_test.json"
    # input_path = f"outputs/{task_dir}/embedding/{args.state_type}/history_embeddings_train_expert.json"
    # input_path = f"outputs/{task_dir}/embedding/{args.state_type}/history_embeddings_filtered.json"

    ids, X = load_ids_and_embeddings_auto(input_path)
    print(ids[0])
    print(f"Loaded: {len(ids)} embeddings, dim={X.shape[1]}")

    if args.normalize:
        X = l2_normalize(X)

    meta = {"cluster_method": args.cluster_method, "task": task_dir, "state_type": args.state_type}

    # ========== KMeans ==========
    if args.cluster_method == "kmeans":
        if args.k is None:
            if args.k_grid:
                ks = parse_klist(args.k_grid)
                k_min, k_max = min(ks), max(ks)
                step = 1
            else:
                k_min, k_max, step = 5, 50, 5

            print(f"[KMeans] Auto choosing k by silhouette in [{k_min}, {k_max}] step={step}")
            sel = choose_k_by_silhouette(
                X,
                k_min=k_min,
                k_max=k_max,
                step=step,
                metric="cosine",
                minibatch=True,
            )
            k = sel["k"]
            meta["silhouette_scores"] = sel["scores"]
            meta["best_silhouette"] = sel["silhouette"]
            print(f"[KMeans] Chosen k={k} (silhouette={sel['silhouette']:.4f})")
        else:
            k = args.k
            print(f"[KMeans] Using specified k={k}")

        model, labels = kmeans_fit_predict(
            X,
            k,
            minibatch=True if args.minibatch or args.k is None else False,
            random_state=args.random_state,
            batch_size=args.batch_size,
            use_gpu=args.use_gpu,
        )
        centroids = getattr(model, "cluster_centers_", None)

        outdir = f"outputs/{task_dir}/cluster/{args.state_type}/kmeans_k{k}_llm"
        export_clusters(outdir, ids, labels, centroids=centroids, meta=meta)
        print(f"[KMeans] Saved to {Path(outdir).resolve()}")

    # ========== OPTICS ==========
    elif args.cluster_method == "optics":
        print(
            f"[OPTICS] Running OPTICS with min_samples={args.min_samples}, "
            f"xi={args.xi}, min_cluster_size={args.min_cluster_size}"
        )

        model, raw_labels = optics_fit_predict(
            X,
            min_samples=args.min_samples,
            xi=args.xi,
            min_cluster_size=args.min_cluster_size,
            metric="cosine",
        )

        labels, centroids = optics_compute_centroids(X, raw_labels)

        valid_mask = labels != -1
        n_clusters = int(centroids.shape[0]) if centroids is not None else 0

        meta["optics_params"] = {
            "min_samples": args.min_samples,
            "xi": args.xi,
            "min_cluster_size": args.min_cluster_size,
        }
        meta["n_clusters"] = n_clusters
        meta["noise_ratio"] = float(np.mean(~valid_mask))

        print(f"[OPTICS] Found {n_clusters} clusters (noise ratio={meta['noise_ratio']:.4f})")

        outdir = f"outputs/{task_dir}/cluster/{args.state_type}/optics_k{n_clusters}_llm"
        export_clusters(outdir, ids, labels, centroids=centroids, meta=meta)
        print(f"[OPTICS] Saved to {Path(outdir).resolve()}")

    # ========== HDBSCAN ==========
    elif args.cluster_method == "hdbscan":
        print(
            f"[HDBSCAN] min_cluster_size={args.hdb_min_cluster_size}, "
            f"min_samples={args.hdb_min_samples}, metric={args.hdb_metric}, "
            f"select={args.hdb_select}, allow_single={args.hdb_allow_single}"
        )

        model, raw_labels = hdbscan_fit_predict(
            X,
            min_cluster_size=args.hdb_min_cluster_size,
            min_samples=args.hdb_min_samples,
            metric=args.hdb_metric,
            cluster_selection_method=args.hdb_select,
            allow_single_cluster=args.hdb_allow_single,
        )
        labels, centroids = hdbscan_compute_centroids(X, raw_labels)

        valid_mask = labels != -1
        n_clusters = int(centroids.shape[0]) if centroids is not None else 0

        meta["hdbscan_params"] = {
            "min_cluster_size": args.hdb_min_cluster_size,
            "min_samples": args.hdb_min_samples,
            "metric": args.hdb_metric,
            "cluster_selection_method": args.hdb_select,
            "allow_single_cluster": args.hdb_allow_single,
        }
        meta["n_clusters"] = n_clusters
        meta["noise_ratio"] = float(np.mean(~valid_mask))

        outdir = f"outputs/{task_dir}/cluster/{args.state_type}/hdbscan_k{n_clusters}_llm"
        export_clusters(outdir, ids, labels, centroids=centroids, meta=meta)
        print(f"[HDBSCAN] Saved to {Path(outdir).resolve()}")
    time2 = time.time()
    print(f"Clustering took {time2 - time1:.2f} seconds")


if __name__ == "__main__":
    main()


"""
Examples:

# P4G
python src/2cluster.py \
    --task P4G \
  --state_type history \
  --normalize \
  --cluster-method kmeans \
  --k 100

# CB
python src/2cluster.py \
  --task CB \
  --state_type history \
  --normalize \
  --cluster-method hdbscan \
  --hdb-min-cluster-size 15 \
  --hdb-metric euclidean \
  --hdb-select eom

# ESC
python src/2cluster.py \
    --task ESC \
    --state_type history \
    --normalize \
    --cluster-method kmeans \
    --k 150
"""
