#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
根据 labels.json 中的聚类结果，把 principles_train_test.json 中的原则
按 cluster 分拆，每个簇一个文件。

⚠️ 输出路径自动设为：
    dirname(labels.json) + "/principles_by_cluster/"

不需要再指定 --outdir
"""

import json
import argparse
import os
from typing import Dict, Any, List


def load_labels(labels_path: str) -> Dict[str, int]:
    """读取 labels.json (JSONL)，返回 id -> cluster 的字典。"""
    id2cluster: Dict[str, int] = {}
    with open(labels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            _id = obj.get("id")
            cluster = obj.get("cluster")
            if _id is None or cluster is None:
                continue
            id2cluster[_id] = int(cluster)
    return id2cluster


def load_principles(principles_path: str) -> List[Dict[str, Any]]:
    with open(principles_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            data = None

    if isinstance(data, list):
        return data

    # Fallback: JSONL
    items: List[Dict[str, Any]] = []
    with open(principles_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                items.append(obj)
    if not items:
        raise ValueError("principles file must be a JSON array or JSONL.")
    return items


def split_by_cluster(principles, id2cluster):
    cluster2items = {}
    missing = 0

    for p in principles:
        _id = p.get("id")
        if _id not in id2cluster:
            missing += 1
            continue

        cluster = id2cluster[_id]

        item = {
            "id": _id,
            "principle_type": p.get("principle_type"),
            "principle_text": p.get("principle_text"),
        }

        cluster2items.setdefault(cluster, []).append(item)

    if missing > 0:
        print(f"[WARN] {missing} principles missing cluster label.")

    return cluster2items


def save_clusters(cluster2items, labels_path: str):
    """
    自动保存到：
        dirname(labels.json)/principles_by_cluster/
    """
    labels_dir = os.path.dirname(labels_path)
    outdir = os.path.join(labels_dir, "principles_by_cluster")

    os.makedirs(outdir, exist_ok=True)
    print(f"Saving clustered principles to: {outdir}")

    for cluster, items in sorted(cluster2items.items(), key=lambda x: x[0]):
        path = os.path.join(outdir, f"cluster_{cluster}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
        print(f"  - wrote {len(items)} items to {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--principles", required=True,
                    help="Path to principles_train_test.json")
    ap.add_argument("--labels", required=True,
                    help="Path to labels.json (JSONL)")
    args = ap.parse_args()

    id2cluster = load_labels(args.labels)
    print(f"Loaded {len(id2cluster)} labels.")

    principles = load_principles(args.principles)
    print(f"Loaded {len(principles)} principles.")

    cluster2items = split_by_cluster(principles, id2cluster)
    print(f"Clusters found: {len(cluster2items)}")

    save_clusters(cluster2items, args.labels)


if __name__ == "__main__":
    main()


'''
python src/4split_principles_by_cluster.py \
    --principles outputs/ESC/principles/micro_principles.jsonl \
  --labels outputs/ESC/cluster/history/kmeans_k150/labels.jsonl
'''
