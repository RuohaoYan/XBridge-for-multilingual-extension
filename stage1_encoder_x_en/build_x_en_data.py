#!/usr/bin/env python3
"""Build x->English JSONL data for encoder-to-LLM Stage 1 warmup.

Output schema:

    {"src_lang": "zho_Hans", "src": "...", "tgt": "English ...", "prompt": "Translate into English:"}

Inputs can be:
1. Parallel text files: --source_file and --english_file.
2. JSON/JSONL: --input_file with source and English target fields.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_SOURCE_FIELDS = ("src", "source", "input", "text", "x")
DEFAULT_TARGET_FIELDS = ("tgt", "target", "english", "en", "translation", "z")
DEFAULT_LANG_FIELDS = ("src_lang", "source_lang", "lang", "language")


def parse_csv(value: str):
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def first_present(record: dict, fields: Iterable[str], default: str = "") -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return default


def iter_json_records(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL line {line_no} must be an object")
                yield row
        return

    if suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict):
            for key in ("data", "train", "records", "items"):
                if isinstance(obj.get(key), list):
                    obj = obj[key]
                    break
        if not isinstance(obj, list):
            raise ValueError("JSON input must be a list or contain data/train/records/items")
        for idx, row in enumerate(obj):
            if not isinstance(row, dict):
                raise ValueError(f"JSON item {idx} must be an object")
            yield row
        return

    raise ValueError(f"Unsupported input suffix: {path.suffix}. Use .jsonl or .json")


def iter_parallel(source_file: Path, english_file: Path, src_lang: str):
    with source_file.open("r", encoding="utf-8") as src_f, english_file.open("r", encoding="utf-8") as en_f:
        for line_no, (src, en) in enumerate(zip(src_f, en_f), start=1):
            yield {
                "src": src.rstrip("\n"),
                "tgt": en.rstrip("\n"),
                "src_lang": src_lang,
                "_line_no": line_no,
            }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build x->English JSONL for mapping_enc2llm training.")
    parser.add_argument("--input_file", default="", help="Optional JSON/JSONL file.")
    parser.add_argument("--source_file", default="", help="Optional source text file.")
    parser.add_argument("--english_file", default="", help="Optional aligned English text file.")
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--src_lang", default="zho_Hans")
    parser.add_argument("--prompt", default="Translate into English:")
    parser.add_argument("--source_fields", default=",".join(DEFAULT_SOURCE_FIELDS))
    parser.add_argument("--target_fields", default=",".join(DEFAULT_TARGET_FIELDS))
    parser.add_argument("--src_lang_fields", default=",".join(DEFAULT_LANG_FIELDS))
    parser.add_argument("--min_chars", type=int, default=1)
    parser.add_argument("--max_chars", type=int, default=0, help="0 disables this filter")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    if args.input_file:
        rows = iter_json_records(Path(args.input_file))
    elif args.source_file and args.english_file:
        rows = iter_parallel(Path(args.source_file), Path(args.english_file), args.src_lang)
    else:
        raise ValueError("Provide either --input_file or --source_file + --english_file")

    source_fields = parse_csv(args.source_fields)
    target_fields = parse_csv(args.target_fields)
    lang_fields = parse_csv(args.src_lang_fields)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in rows:
            src = normalize_text(first_present(row, source_fields))
            tgt = normalize_text(first_present(row, target_fields))
            src_lang = first_present(row, lang_fields, args.src_lang)

            if len(src) < args.min_chars or len(tgt) < args.min_chars:
                skipped += 1
                continue
            if args.max_chars > 0 and (len(src) > args.max_chars or len(tgt) > args.max_chars):
                skipped += 1
                continue

            record = {"src_lang": src_lang, "src": src, "tgt": tgt, "prompt": args.prompt}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if args.limit > 0 and written >= args.limit:
                break

    print(f"Wrote {written} examples to {output_path}")
    if skipped:
        print(f"Skipped {skipped} examples by length filters")


if __name__ == "__main__":
    main()
