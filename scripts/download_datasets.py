#!/usr/bin/env python3
"""Download evaluation datasets for XBridge paper reproduction."""

import csv
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

MGSM_LANGS = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]
MGSM_BASE = "https://raw.githubusercontent.com/google-research/url-nlp/main/mgsm"

FLORES_LANGS = {
    "en": "eng", "zh": "zho_simpl", "es": "spa", "fr": "fra",
    "th": "tha", "sw": "swh", "ja": "jpn", "bn": "ben",
    "de": "deu", "ru": "rus",
}
FLORES_TARBALL = "https://dl.fbaipublicfiles.com/flores101/dataset/flores101_dataset.tar.gz"
FLORES_BASE = (
    "https://raw.githubusercontent.com/facebookresearch/flores/"
    "main/previous_releases/flores101_dataset/devtest"
)


def fetch_url(url, out_path, timeout=120):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)


def download_mgsm():
    out_dir = os.path.join(DATA_DIR, "mgsm")
    os.makedirs(out_dir, exist_ok=True)
    for lang in MGSM_LANGS:
        out_path = os.path.join(out_dir, f"mgsm_{lang}.json")
        if os.path.exists(out_path):
            print(f"[skip] {out_path}")
            continue
        tsv_url = f"{MGSM_BASE}/mgsm_{lang}.tsv"
        tsv_path = os.path.join(out_dir, f"mgsm_{lang}.tsv")
        print(f"[download] MGSM {lang}...")
        fetch_url(tsv_url, tsv_path)
        records = []
        with open(tsv_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line:
                    continue
                if "\t" in line:
                    question, answer_num = line.split("\t", 1)
                else:
                    parts = line.rsplit(" ", 1)
                    question, answer_num = parts[0], parts[1]
                records.append({
                    "question": question.strip(),
                    "answer": answer_num.strip(),
                })
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        os.remove(tsv_path)
        print(f"  -> {len(records)} samples saved to {out_path}")


def download_flores():
    import shutil
    import tarfile
    import tempfile

    out_dir = os.path.join(DATA_DIR, "flores101")
    os.makedirs(out_dir, exist_ok=True)

    needed = [f"{code}.devtest" for code in FLORES_LANGS.values()]
    if all(os.path.exists(os.path.join(out_dir, f)) for f in needed):
        print("[skip] FLORES-101 already present")
        return

    for lang, flores_code in FLORES_LANGS.items():
        out_path = os.path.join(out_dir, f"{flores_code}.devtest")
        if os.path.exists(out_path):
            continue
        url = f"{FLORES_BASE}/{flores_code}.devtest"
        print(f"[download] FLORES-101 {lang} ({flores_code}) from GitHub...")
        try:
            fetch_url(url, out_path, timeout=30)
            continue
        except Exception as e:
            print(f"  GitHub failed ({e}), will try official tarball...")
            if os.path.exists(out_path):
                os.remove(out_path)
            break

    if all(os.path.exists(os.path.join(out_dir, f)) for f in needed):
        return

    print("[download] FLORES-101 official tarball...")
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
        tarball_path = tmp.name
    try:
        fetch_url(FLORES_TARBALL, tarball_path, timeout=300)
        with tarfile.open(tarball_path, "r:gz") as tar:
            tar.extractall(path=out_dir)
        devtest_dir = os.path.join(out_dir, "devtest")
        if os.path.isdir(devtest_dir):
            for flores_code in FLORES_LANGS.values():
                src = os.path.join(devtest_dir, f"{flores_code}.devtest")
                dst = os.path.join(out_dir, f"{flores_code}.devtest")
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copy2(src, dst)
        for flores_code in FLORES_LANGS.values():
            out_path = os.path.join(out_dir, f"{flores_code}.devtest")
            if not os.path.exists(out_path):
                raise FileNotFoundError(f"Missing {out_path}")
            with open(out_path, "r", encoding="utf-8") as f:
                n = sum(1 for _ in f)
            print(f"  -> {n} lines: {out_path}")
    finally:
        if os.path.exists(tarball_path):
            os.remove(tarball_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mgsm", action="store_true", help="Download MGSM")
    parser.add_argument("--flores", action="store_true", help="Download FLORES-101")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    args = parser.parse_args()
    if args.all or args.mgsm:
        download_mgsm()
    if args.all or args.flores:
        download_flores()
    if not (args.all or args.mgsm or args.flores):
        parser.print_help()
        print("\nFor large x→en training data (encoder-only), see:")
        print("  python scripts/download_parallel_xen.py --sources nllb --max_samples_per_lang 200000")
