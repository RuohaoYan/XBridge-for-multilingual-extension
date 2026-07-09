#!/usr/bin/env python3
"""Download large x→en parallel data for encoder-only XBridge training.

Only the English side of bitext is used as LLM label; a third language (if any) is ignored.

Sources:
  nllb   - Meta/AllenAI mined bitext (~450GB total; streamed per language pair from GCS)
  seed   - NLLB-Seed (~6k high-quality Wikipedia sentences, 39 langs)
  flores - local FLORES dev/devtest (small, for smoke tests)

Output: JSONL compatible with train_encoder_only.py
  {"task":"translate","src_lang":"zho_Hans","src":"...","tgt":"..."}
"""

import argparse
import gzip
import io
import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from typing import Iterable, Iterator, List, Optional, Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ISO-ish keys -> NLLB codes used by tokenizer_mt.src_lang
LANG_TO_NLLB = {
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

FLORES_FILE = {
    "zh": "zho_simpl",
    "de": "deu",
    "es": "spa",
    "fr": "fra",
    "ja": "jpn",
    "ru": "rus",
    "th": "tha",
    "sw": "swh",
    "bn": "ben",
}

NLLB_GCS = "https://storage.googleapis.com/allennlp-data-bucket/nllb/{pair}.gz"
NLLB_SEED_URL = "https://dl.fbaipublicfiles.com/nllb/NLLB-Seed.zip"


def fetch_url(url: str, out_path: str, timeout: int = 600):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    with open(out_path, "wb") as f:
        f.write(data)
    return out_path


def stream_gz_tsv(url: str, timeout: int = 600) -> Iterator[Tuple[str, str]]:
    """Stream tab-separated parallel lines from a remote .gz file."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with gzip.GzipFile(fileobj=resp) as gz:
            for raw in gz:
                line = raw.decode("utf-8", errors="replace").rstrip("\n")
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) < 2:
                    continue
                yield parts[0].strip(), parts[1].strip()


def write_jsonl(records: Iterable[dict], out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def download_nllb_xen(
    langs: List[str],
    max_samples_per_lang: int,
    cache_dir: str,
) -> Iterator[dict]:
    os.makedirs(cache_dir, exist_ok=True)
    for lang in langs:
        nllb = LANG_TO_NLLB[lang]
        pair = f"{nllb}-eng_Latn"
        url = NLLB_GCS.format(pair=pair)
        local_gz = os.path.join(cache_dir, f"{pair}.gz")
        print(f"[nllb] {lang} ({pair}) <- {url}")

        if not os.path.isfile(local_gz):
            try:
                fetch_url(url, local_gz)
            except Exception as exc:
                print(f"  download failed: {exc}")
                continue
        else:
            print(f"  using cached {local_gz}")

        count = 0
        with gzip.open(local_gz, "rt", encoding="utf-8", errors="replace") as gz:
            for line in gz:
                if max_samples_per_lang > 0 and count >= max_samples_per_lang:
                    break
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                src, tgt = parts[0].strip(), parts[1].strip()
                if not src or not tgt:
                    continue
                yield {
                    "task": "translate",
                    "src_lang": nllb,
                    "src": src,
                    "tgt": tgt,
                }
                count += 1
        print(f"  -> {count} pairs")


def _read_text_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def download_seed_xen(langs: List[str], cache_dir: str) -> Iterator[dict]:
    """NLLB-Seed: line-aligned monolingual files; use English + each target lang."""
    os.makedirs(cache_dir, exist_ok=True)
    zip_path = os.path.join(cache_dir, "NLLB-Seed.zip")
    extract_dir = os.path.join(cache_dir, "NLLB-Seed")

    if not os.path.isdir(extract_dir):
        if not os.path.isfile(zip_path):
            print(f"[seed] downloading {NLLB_SEED_URL}")
            fetch_url(NLLB_SEED_URL, zip_path, timeout=900)
        print(f"[seed] extracting -> {extract_dir}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)

    # Seed layout: {flores_code}.txt or similar; find eng + lang files
    def find_file(code: str) -> Optional[str]:
        for root, _, files in os.walk(extract_dir):
            for name in files:
                if name == f"{code}.txt" or name == f"{code}.dev" or name.endswith(f"/{code}.txt"):
                    return os.path.join(root, name)
                if name.startswith(code) and name.endswith(".txt"):
                    return os.path.join(root, name)
        return None

    eng_path = find_file("eng") or find_file("eng_Latn")
    if not eng_path:
        raise FileNotFoundError("Could not find English file in NLLB-Seed archive")
    en_lines = _read_text_lines(eng_path)

    seed_flores = {
        "zh": "zho_simpl",
        "de": "deu",
        "es": "spa",
        "fr": "fra",
        "ja": "jpn",
        "ru": "rus",
        "th": "tha",
        "sw": "swh",
        "bn": "ben",
    }

    for lang in langs:
        code = seed_flores.get(lang)
        nllb = LANG_TO_NLLB[lang]
        src_path = find_file(code) if code else None
        if not src_path:
            print(f"[seed] skip {lang}: file not in seed set")
            continue
        src_lines = _read_text_lines(src_path)
        n = min(len(src_lines), len(en_lines))
        print(f"[seed] {lang}: {n} pairs")
        for i in range(n):
            if not src_lines[i].strip() or not en_lines[i].strip():
                continue
            yield {
                "task": "translate",
                "src_lang": nllb,
                "src": src_lines[i].strip(),
                "tgt": en_lines[i].strip(),
            }


def flores_xen_local(langs: List[str], flores_dir: str, split: str) -> Iterator[dict]:
    sub = "dev" if split == "dev" else "devtest"
    eng_path = os.path.join(flores_dir, sub, "eng.dev" if split == "dev" else "eng.devtest")
    if not os.path.isfile(eng_path):
        eng_path = os.path.join(flores_dir, "eng.devtest")
    en_lines = _read_text_lines(eng_path)

    for lang in langs:
        code = FLORES_FILE[lang]
        src_path = os.path.join(flores_dir, sub, f"{code}.dev" if split == "dev" else f"{code}.devtest")
        if not os.path.isfile(src_path):
            src_path = os.path.join(flores_dir, f"{code}.devtest")
        if not os.path.isfile(src_path):
            print(f"[flores] skip {lang}: missing {src_path}")
            continue
        src_lines = _read_text_lines(src_path)
        n = min(len(src_lines), len(en_lines))
        print(f"[flores] {lang}: {n} pairs")
        for i in range(n):
            yield {
                "task": "translate",
                "src_lang": LANG_TO_NLLB[lang],
                "src": src_lines[i].strip(),
                "tgt": en_lines[i].strip(),
            }


def trilingual_tsv_xen(
    tsv_path: str,
    src_lang: str,
    src_col: int = 0,
    en_col: int = 1,
) -> Iterator[dict]:
    """Read arbitrary trilingual TSV; keep src_col and en_col only."""
    nllb = LANG_TO_NLLB.get(src_lang, src_lang)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if max(src_col, en_col) >= len(parts):
                continue
            src, tgt = parts[src_col].strip(), parts[en_col].strip()
            if not src or not tgt:
                continue
            yield {
                "task": "translate",
                "src_lang": nllb,
                "src": src,
                "tgt": tgt,
            }


def main():
    parser = argparse.ArgumentParser(description="Download/build large x→en JSONL for encoder-only training")
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/train_nllb_xen.jsonl"),
    )
    parser.add_argument(
        "--sources",
        default="nllb",
        help="Comma-separated: nllb, seed, flores",
    )
    parser.add_argument(
        "--langs",
        default="zh,de,es,fr,ja,ru,th,sw,bn",
        help="Language keys (exclude en)",
    )
    parser.add_argument(
        "--max_samples_per_lang",
        type=int,
        default=200_000,
        help="Cap per language for nllb source (0 = no cap). Full pair can be >10M lines.",
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(ROOT, "data/parallel_cache"),
    )
    parser.add_argument(
        "--flores_dir",
        default=os.path.join(ROOT, "data/flores101"),
    )
    parser.add_argument(
        "--flores_split",
        default="dev",
        choices=("dev", "devtest"),
    )
    parser.add_argument(
        "--trilingual_tsv",
        default="",
        help="Optional local trilingual TSV; uses --src_col / --en_col, ignores 3rd col",
    )
    parser.add_argument("--src_col", type=int, default=0)
    parser.add_argument("--en_col", type=int, default=1)
    parser.add_argument("--tsv_src_lang", default="zh")
    args = parser.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    sources = [x.strip() for x in args.sources.split(",") if x.strip()]

    def all_records() -> Iterator[dict]:
        if args.trilingual_tsv:
            print(f"[tsv] {args.trilingual_tsv} cols=({args.src_col},{args.en_col})")
            yield from trilingual_tsv_xen(
                args.trilingual_tsv, args.tsv_src_lang, args.src_col, args.en_col
            )
        for src in sources:
            if src == "nllb":
                yield from download_nllb_xen(langs, args.max_samples_per_lang, args.cache_dir)
            elif src == "seed":
                yield from download_seed_xen(langs, args.cache_dir)
            elif src == "flores":
                yield from flores_xen_local(langs, args.flores_dir, args.flores_split)
            else:
                raise ValueError(f"Unknown source: {src}")

    n = write_jsonl(all_records(), args.output)
    print(f"Wrote {n} samples -> {args.output}")


if __name__ == "__main__":
    main()
