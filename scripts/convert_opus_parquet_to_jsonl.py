#!/usr/bin/env python3
"""Convert local OPUS-100 en-zh parquet to encoder-only training JSONL."""

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def convert(parquet_path: str, output: str, max_samples: int = 0):
    import pyarrow.parquet as pq

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    pf = pq.ParquetFile(parquet_path)
    print(f"[input] {parquet_path} ({pf.metadata.num_rows:,} rows)")
    print(f"[output] {output}")
    if max_samples > 0:
        print(f"[limit] {max_samples:,} samples")

    n = 0
    t0 = time.time()
    with open(output, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=8192):
            col = batch.column("translation")
            for i in range(batch.num_rows):
                if max_samples > 0 and n >= max_samples:
                    break
                item = col[i].as_py()
                zh = (item.get("zh") or "").strip()
                en = (item.get("en") or "").strip()
                if not zh or not en:
                    continue
                rec = {
                    "task": "translate",
                    "src_lang": "zho_Hans",
                    "src": zh,
                    "tgt": en,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                if n % 20000 == 0:
                    elapsed = max(time.time() - t0, 1e-6)
                    sys.stdout.write(f"\r  [convert] {n:,} / {max_samples or pf.metadata.num_rows:,}  {n/elapsed:,.0f}/s")
                    sys.stdout.flush()
            if max_samples > 0 and n >= max_samples:
                break

    elapsed = max(time.time() - t0, 1e-6)
    print(f"\r  [convert] {n:,} samples done in {elapsed:.1f}s ({n/elapsed:,.0f}/s)")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--parquet",
        default=os.path.join(ROOT, "data/Helsinki-NLP_opus-100/en-zh/train-00000-of-00001.parquet"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/opus100_zh_en_100k.jsonl"),
    )
    parser.add_argument("--max_samples", type=int, default=100000)
    args = parser.parse_args()

    if not os.path.isfile(args.parquet):
        raise SystemExit(f"Missing parquet: {args.parquet}")

    n = convert(args.parquet, args.output, args.max_samples)
    print(f"Wrote {n:,} samples -> {args.output}")


if __name__ == "__main__":
    main()
