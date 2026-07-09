#!/usr/bin/env python3
"""Extract English sentences and build exact-repetition JSONL for encoder-only training.

Training sequence (train_encoder_only.py collate, no prompt):
  Enc(src) + boundary + tgt
  tgt = src  (same English sentence, pure causal continuation)
"""

import argparse
import json
import os
import re
from typing import Iterable, List, Set

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def iter_english_from_jsonl(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            tgt = row.get("tgt") or row.get("en") or ""
            if tgt:
                yield normalize_sentence(tgt)


def build_repeat_records(
    sentences: Iterable[str],
    dedupe: bool = True,
    min_chars: int = 3,
    max_chars: int = 512,
) -> List[dict]:
    seen: Set[str] = set()
    records: List[dict] = []
    skipped = {"empty": 0, "short": 0, "long": 0, "dup": 0}

    for sent in sentences:
        if not sent:
            skipped["empty"] += 1
            continue
        if len(sent) < min_chars:
            skipped["short"] += 1
            continue
        if len(sent) > max_chars:
            skipped["long"] += 1
            continue
        key = sent.casefold()
        if dedupe:
            if key in seen:
                skipped["dup"] += 1
                continue
            seen.add(key)

        records.append(
            {
                "task": "repeat",
                "src_lang": "eng_Latn",
                "src": sent,
                "tgt": sent,
            }
        )
    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "data/encoder_only/opus100_zh_en_100k.jsonl"),
        help="Source JSONL with English in `tgt` field.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/repeat_en_100k.jsonl"),
    )
    parser.add_argument("--no_dedupe", action="store_true")
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=512)
    args = parser.parse_args()

    records, skipped = build_repeat_records(
        iter_english_from_jsonl(args.input),
        dedupe=not args.no_dedupe,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Wrote {len(records)} repeat samples")
    print("Skipped:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")
    if records:
        print("Sample:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
