#!/usr/bin/env python3
"""Download OPUS-100 en-zh (or other en-xx) and export x→en JSONL for encoder-only training."""

import argparse
import json
import os
import sys
import tarfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from download_utils import download_url

# OPUS-100 config name -> NLLB src_lang for tokenizer_mt
OPUS_TO_NLLB = {
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

OPUS_URL = "https://object.pouta.csc.fi/OPUS-100/v1.0/opus-100-corpus-en-{lang}-v1.0.tar.gz"
EXPECTED_BYTES = {"zh": 83265500}  # en-zh v1.0 from OPUS mirror


def validate_tar_gz(path: str, lang: str) -> bool:
    try:
        with tarfile.open(path, "r:gz") as tar:
            tar.getmembers()
        exp = EXPECTED_BYTES.get(lang)
        if exp and os.path.getsize(path) < exp * 0.99:
            return False
        return True
    except (EOFError, tarfile.ReadError, OSError):
        return False


def find_pair_files(extract_dir: str, lang: str, split: str):
    """Return (en_path, xx_path) for OPUS-100 en-xx split."""
    base = os.path.join(extract_dir, "opus-100-corpus", "v1.0", "supervised", f"en-{lang}")
    en_path = os.path.join(base, f"opus.en-{lang}-{split}.en")
    xx_path = os.path.join(base, f"opus.en-{lang}-{split}.{lang}")
    if os.path.isfile(en_path) and os.path.isfile(xx_path):
        return en_path, xx_path
    # walk fallback
    for root, _, files in os.walk(extract_dir):
        if f"opus.en-{lang}-{split}.en" in files:
            return (
                os.path.join(root, f"opus.en-{lang}-{split}.en"),
                os.path.join(root, f"opus.en-{lang}-{split}.{lang}"),
            )
    raise FileNotFoundError(f"Cannot find en-{lang} {split} files under {extract_dir}")


def export_xen_jsonl(en_path: str, xx_path: str, src_lang_nllb: str, out_path: str, max_samples: int = 0):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    t0 = time.time()
    with open(en_path, "r", encoding="utf-8") as fen, open(xx_path, "r", encoding="utf-8") as fxx, open(
        out_path, "w", encoding="utf-8"
    ) as fout:
        for en_line, xx_line in zip(fen, fxx):
            if max_samples > 0 and n >= max_samples:
                break
            en = en_line.rstrip("\n").strip()
            xx = xx_line.rstrip("\n").strip()
            if not en or not xx:
                continue
            rec = {"task": "translate", "src_lang": src_lang_nllb, "src": xx, "tgt": en}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
            if n % 50000 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                sys.stdout.write(f"\r  [convert] {n:,} pairs  {n/elapsed:,.0f} pairs/s")
                sys.stdout.flush()
    if n:
        sys.stdout.write(f"\r  [convert] {n:,} pairs done\n")
        sys.stdout.flush()
    return n


def main():
    parser = argparse.ArgumentParser(description="Download OPUS-100 en-xx -> x→en JSONL")
    parser.add_argument("--lang", default="zh", help="Non-English language code in OPUS-100 (e.g. zh, de, ja)")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/opus100_zh_en_train.jsonl"),
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(ROOT, "data/parallel_cache/opus100"),
    )
    parser.add_argument("--split", default="train", choices=("train", "dev", "test"))
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all lines in split")
    parser.add_argument("--force", action="store_true", help="Re-download even if cache exists")
    args = parser.parse_args()

    if args.lang not in OPUS_TO_NLLB:
        raise SystemExit(f"Unsupported lang {args.lang}. Supported: {', '.join(OPUS_TO_NLLB)}")

    os.makedirs(args.cache_dir, exist_ok=True)
    tar_path = os.path.join(args.cache_dir, f"opus-100-corpus-en-{args.lang}-v1.0.tar.gz")
    extract_dir = os.path.join(args.cache_dir, f"en-{args.lang}")

    if args.force:
        if os.path.isfile(tar_path):
            os.remove(tar_path)
        if os.path.isdir(extract_dir):
            import shutil
            shutil.rmtree(extract_dir)

    need_download = not os.path.isfile(tar_path) or not validate_tar_gz(tar_path, args.lang)
    if need_download:
        if os.path.isfile(tar_path):
            print(f"[warn] corrupt/incomplete tar, re-downloading: {tar_path}")
            os.remove(tar_path)
        url = OPUS_URL.format(lang=args.lang)
        download_url(url, tar_path, label=f"OPUS-100 en-{args.lang}")

    if not os.path.isdir(extract_dir):
        print(f"[extract] -> {extract_dir}")
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(tar_path, "r:gz") as tar:
            members = tar.getmembers()
            total = len(members)
            for i, m in enumerate(members, 1):
                tar.extract(m, extract_dir)
                sys.stdout.write(f"\r  extracting {i}/{total}")
                sys.stdout.flush()
            sys.stdout.write("\n")

    en_path, xx_path = find_pair_files(extract_dir, args.lang, args.split)
    print(f"[convert] {args.split}: {xx_path} + {en_path}")
    n = export_xen_jsonl(en_path, xx_path, OPUS_TO_NLLB[args.lang], args.output, args.max_samples)
    print(f"Wrote {n} zh→en samples -> {args.output}")


if __name__ == "__main__":
    main()
