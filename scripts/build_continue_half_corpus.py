#!/usr/bin/env python3
"""Build half-sentence continuation corpus for encoder-only training.

Split each English sentence near the middle (word boundary):
  src = first half  ->  Enc(src) + boundary + tgt
  tgt = second half (causal continuation, no prompt)
  orig_tgt = full original sentence (kept for comparison)
"""

import argparse
import json
import os
import re
from typing import Dict, Iterable, List, Set, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def normalize_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_half(sentence: str) -> Tuple[str, str]:
    """Split at the space closest to the midpoint."""
    s = sentence.strip()
    if not s:
        return "", ""
    spaces = [i for i, c in enumerate(s) if c == " "]
    if not spaces:
        return "", ""
    mid = len(s) / 2
    cut = min(spaces, key=lambda i: abs(i - mid))
    return s[:cut].strip(), s[cut:].strip()


def iter_records(
    path: str,
    dedupe: bool = True,
    min_orig_chars: int = 8,
    min_part_chars: int = 3,
    max_orig_chars: int = 512,
    limit: int = 0,
) -> Tuple[List[dict], Dict[str, int]]:
    seen: Set[str] = set()
    records: List[dict] = []
    skipped = {
        "empty": 0,
        "short_orig": 0,
        "long_orig": 0,
        "short_part": 0,
        "dup": 0,
    }

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            orig = normalize_sentence(row.get("tgt") or row.get("en") or "")
            if not orig:
                skipped["empty"] += 1
                continue
            if len(orig) < min_orig_chars:
                skipped["short_orig"] += 1
                continue
            if len(orig) > max_orig_chars:
                skipped["long_orig"] += 1
                continue

            key = orig.casefold()
            if dedupe:
                if key in seen:
                    skipped["dup"] += 1
                    continue
                seen.add(key)

            src, tgt = split_half(orig)
            if len(src) < min_part_chars or len(tgt) < min_part_chars:
                skipped["short_part"] += 1
                continue

            records.append(
                {
                    "task": "continue",
                    "src_lang": "eng_Latn",
                    "src": src,
                    "tgt": tgt,
                    "orig_tgt": orig,
                }
            )
            if limit and len(records) >= limit:
                break

    return records, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "data/encoder_only/opus100_zh_en_100k.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/continue_en_half_50k.jsonl"),
    )
    parser.add_argument("--limit", type=int, default=50000)
    parser.add_argument("--min_orig_chars", type=int, default=8)
    parser.add_argument("--min_part_chars", type=int, default=3)
    parser.add_argument("--max_orig_chars", type=int, default=512)
    parser.add_argument("--no_dedupe", action="store_true")
    args = parser.parse_args()

    records, skipped = iter_records(
        args.input,
        dedupe=not args.no_dedupe,
        min_orig_chars=args.min_orig_chars,
        min_part_chars=args.min_part_chars,
        max_orig_chars=args.max_orig_chars,
        limit=args.limit,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Wrote {len(records)} samples (limit={args.limit})")
    print("Skipped:")
    for k, v in skipped.items():
        print(f"  {k}: {v}")
    if records:
        print("Sample:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
