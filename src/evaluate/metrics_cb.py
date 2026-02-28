#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from metrics import compute_cb_metrics, load_evaluation_results
import json


def main():
    parser = argparse.ArgumentParser("CB Metrics")
    parser.add_argument("--input", type=str, required=True, help="Path to result JSON/JSONL")
    parser.add_argument("--max_turns", type=int, default=10)
    parser.add_argument("--save_json", type=str, default=None)
    args = parser.parse_args()

    data = load_evaluation_results(args.input)
    metrics = compute_cb_metrics(data, max_turns=args.max_turns)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
