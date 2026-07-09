#!/usr/bin/env python3
"""Build JSONL training data for encoder-only XBridge (NLLB enc + LLM)."""

import argparse
import json
import os
from typing import Dict, List, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FLORES2NLLB = {
    "eng": "eng_Latn",
    "zho_simpl": "zho_Hans",
    "deu": "deu_Latn",
    "spa": "spa_Latn",
    "fra": "fra_Latn",
    "jpn": "jpn_Jpan",
    "rus": "rus_Cyrl",
    "tha": "tha_Thai",
    "swh": "swh_Latn",
    "ben": "ben_Beng",
}

MGSM_LANG = {
    "en": "eng_Latn",
    "zh": "zho_Hans",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "sw": "swh_Latn",
    "bn": "ben_Beng",
}

MGSM_INSTRUCTION = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{question}\n\n"
    "### Response: Let's think step by step."
)


def read_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def flores_xen_pairs(flores_dir: str, split: str = "dev") -> List[dict]:
    """Parallel (x, en) pairs from FLORES for x→en translation."""
    records = []
    eng_path = os.path.join(flores_dir, split, "eng.dev" if split == "dev" else "eng.devtest")
    en_lines = read_lines(eng_path)
    for code, nllb in FLORES2NLLB.items():
        if code == "eng":
            continue
        src_path = os.path.join(
            flores_dir,
            split,
            f"{code}.dev" if split == "dev" else f"{code}.devtest",
        )
        if not os.path.isfile(src_path):
            continue
        src_lines = read_lines(src_path)
        n = min(len(src_lines), len(en_lines))
        for i in range(n):
            records.append(
                {
                    "task": "translate",
                    "src_lang": nllb,
                    "src": src_lines[i],
                    "tgt": en_lines[i],
                }
            )
    return records


def mgsm_records(mgsm_dir: str, langs: Tuple[str, ...]) -> List[dict]:
    """MGSM (question in x, English chain-of-thought style target)."""
    records = []
    for lang in langs:
        if lang == "en":
            continue
        path = os.path.join(mgsm_dir, f"mgsm_{lang}.json")
        if not os.path.isfile(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        nllb = MGSM_LANG[lang]
        for item in items:
            q = item["question"]
            records.append(
                {
                    "task": "mgsm",
                    "src_lang": nllb,
                    "src": q,
                    "prompt": MGSM_INSTRUCTION.format(question=q),
                    "tgt": item.get("answer_en") or item.get("answer", ""),
                }
            )
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/train.jsonl"),
    )
    parser.add_argument(
        "--flores_dir",
        default=os.path.join(ROOT, "data/flores101"),
    )
    parser.add_argument(
        "--mgsm_dir",
        default=os.path.join(ROOT, "data/mgsm"),
    )
    parser.add_argument(
        "--flores_split",
        default="dev",
        choices=("dev", "devtest"),
    )
    parser.add_argument(
        "--include_mgsm",
        action="store_true",
        help="Append MGSM samples (stage-2 style); default is FLORES x→en only.",
    )
    parser.add_argument(
        "--mgsm_langs",
        default="zh,de,es,fr,ja,ru,th,sw,bn",
    )
    args = parser.parse_args()

    records = flores_xen_pairs(args.flores_dir, args.flores_split)
    if args.include_mgsm:
        langs = tuple(x.strip() for x in args.mgsm_langs.split(",") if x.strip())
        records.extend(mgsm_records(args.mgsm_dir, langs))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    tasks: Dict[str, int] = {}
    for rec in records:
        tasks[rec["task"]] = tasks.get(rec["task"], 0) + 1
    print(f"Wrote {len(records)} samples -> {args.output}")
    for task, count in sorted(tasks.items()):
        print(f"  {task}: {count}")


if __name__ == "__main__":
    main()
