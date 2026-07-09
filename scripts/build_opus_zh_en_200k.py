#!/usr/bin/env python3
"""Export 200k zh->en pairs from the local OPUS-100 en-zh train parquet.

Same source as the existing opus100_zh_en_100k.jsonl (translation.{en,zh}),
extended to 200k with dedup + light sanity filtering.
"""
import json
import os

import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "data/Helsinki-NLP_opus-100/en-zh/train-00000-of-00001.parquet")
OUT = os.path.join(ROOT, "data/encoder_only/opus100_zh_en_200k.jsonl")
TARGET = 200_000


def ok(zh: str, en: str) -> bool:
    if not zh or not en:
        return False
    if len(zh) > 300 or len(en) > 400:
        return False
    # drop degenerate repetition noise (very low unique-token ratio)
    toks = en.split()
    if len(toks) >= 8 and len(set(toks)) / len(toks) < 0.4:
        return False
    return True


def main():
    pf = pq.ParquetFile(SRC)
    print(f"[input] {SRC} ({pf.metadata.num_rows:,} rows)")
    seen = set()
    n = 0
    with open(OUT, "w", encoding="utf-8") as fout:
        for batch in pf.iter_batches(batch_size=10000, columns=["translation"]):
            for item in batch.to_pylist():
                t = item["translation"]
                zh = (t.get("zh") or "").strip()
                en = (t.get("en") or "").strip()
                if not ok(zh, en):
                    continue
                key = (zh, en)
                if key in seen:
                    continue
                seen.add(key)
                fout.write(json.dumps(
                    {"task": "translate", "src_lang": "zho_Hans", "src": zh, "tgt": en},
                    ensure_ascii=False) + "\n")
                n += 1
                if n >= TARGET:
                    print(f"[done] wrote {n:,} pairs -> {OUT}")
                    return
    print(f"[warn] source exhausted at {n:,} pairs (< {TARGET:,}) -> {OUT}")


if __name__ == "__main__":
    main()
