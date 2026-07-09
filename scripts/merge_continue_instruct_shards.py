#!/usr/bin/env python3
"""Merge shard jsonl files into one continue_en_half_50k_instruct.jsonl."""

import argparse
import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        default=os.path.join(
            ROOT, "data/encoder_only/continue_en_half_50k_instruct.shard*.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/continue_en_half_50k_instruct.jsonl"),
    )
    args = parser.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        raise SystemExit(f"No shard files match: {args.pattern}")

    seen = set()
    written = 0
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for path in files:
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    row = json.loads(line)
                    key = row["src"]
                    if key in seen:
                        continue
                    seen.add(key)
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    written += 1

    print(f"Merged {len(files)} shards -> {args.output}")
    print(f"Total unique rows: {written}")
    for path in files:
        print(f"  {path}")


if __name__ == "__main__":
    main()
