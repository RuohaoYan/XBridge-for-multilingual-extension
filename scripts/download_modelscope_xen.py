#!/usr/bin/env python3
"""Download zh→en parallel data via ModelScope (国内镜像，速度通常更快).

Supported datasets:
  wmt   - damo/WMT-Chinese-to-English (~25M pairs, 5.9GB CSV)
  nllb  - modelscope/nllb, subset zho_Hans-eng_Latn (mined bitext, per-pair gzip)
  opus  - try snapshot Helsinki-NLP/opus-100 en-zh (~1M pairs, ~83MB tar.gz)

Output JSONL for train_encoder_only.py:
  {"task":"translate","src_lang":"zho_Hans","src":"...","tgt":"..."}

Requires: pip install modelscope
"""

import argparse
import csv
import gzip
import json
import os
import subprocess
import sys
import tarfile
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)

WMT_REPO = "damo/WMT-Chinese-to-English-Machine-Translation-Training-Corpus"
WMT_CSV = "wmt_zh_en_training_corpus.csv"
NLLB_REPO = "modelscope/nllb"
NLLB_PAIR = "zho_Hans-eng_Latn"
OPUS_REPO = "Helsinki-NLP/opus-100"
OPUS_PARQUET = "en-zh/train-00000-of-00001.parquet"


def export_jsonl(records, out_path: str) -> int:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return n


def convert_wmt_csv(csv_path: str, max_samples: int = 0):
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        # expect columns like zh, en or similar
        for i, row in enumerate(reader):
            if max_samples > 0 and i >= max_samples:
                break
            if len(row) < 2:
                continue
            zh, en = row[0].strip(), row[1].strip()
            if not zh or not en:
                continue
            yield {
                "task": "translate",
                "src_lang": "zho_Hans",
                "src": zh,
                "tgt": en,
            }


def download_wmt(cache_dir: str) -> str:
    from modelscope.hub.file_download import dataset_file_download

    print(f"[modelscope/wmt] {WMT_REPO}/{WMT_CSV}")
    return dataset_file_download(WMT_REPO, WMT_CSV, cache_dir=cache_dir)


def download_nllb_pair(cache_dir: str) -> str:
    from modelscope.hub.snapshot_download import snapshot_download

    print(f"[modelscope/nllb] {NLLB_REPO} ({NLLB_PAIR})")
    root = snapshot_download(NLLB_REPO, repo_type="dataset", cache_dir=cache_dir)
    # find {pair}.gz in tree
    target = f"{NLLB_PAIR}.gz"
    for dirpath, _, files in os.walk(root):
        if target in files:
            return os.path.join(dirpath, target)
    raise FileNotFoundError(f"{target} not found under {root}")


def convert_nllb_gz(gz_path: str, max_samples: int = 0):
    n = 0
    with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as gz:
        for line in gz:
            if max_samples > 0 and n >= max_samples:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            src, tgt = parts[0].strip(), parts[1].strip()
            if not src or not tgt:
                continue
            yield {"task": "translate", "src_lang": "zho_Hans", "src": src, "tgt": tgt}
            n += 1


def download_opus_enzh_parquet(cache_dir: str) -> str:
    from modelscope.hub.file_download import dataset_file_download

    print(f"[modelscope/opus] {OPUS_REPO}/{OPUS_PARQUET}")
    print("  (ModelScope 会显示实时下载进度条)")
    return dataset_file_download(OPUS_REPO, OPUS_PARQUET, cache_dir=cache_dir)


def convert_opus_parquet(parquet_path: str, max_samples: int = 0):
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(parquet_path)
    n = 0
    t0 = time.time()
    for batch in pf.iter_batches(batch_size=8192):
        col = batch.column("translation")
        for i in range(batch.num_rows):
            if max_samples > 0 and n >= max_samples:
                return
            item = col[i].as_py()
            zh = (item.get("zh") or "").strip()
            en = (item.get("en") or "").strip()
            if not zh or not en:
                continue
            yield {"task": "translate", "src_lang": "zho_Hans", "src": zh, "tgt": en}
            n += 1
            if n % 50000 == 0:
                elapsed = max(time.time() - t0, 1e-6)
                sys.stdout.write(f"\r  [convert] {n:,} pairs  {n/elapsed:,.0f} pairs/s")
                sys.stdout.flush()
    if n:
        sys.stdout.write(f"\r  [convert] {n:,} pairs done\n")
        sys.stdout.flush()


def download_opus_enzh(output: str, cache_dir: str, max_samples: int) -> str:
    """Prefer ModelScope single en-zh parquet (~100MB); fallback to direct OPUS tar."""
    try:
        parquet_path = download_opus_enzh_parquet(cache_dir)
        n = export_jsonl(convert_opus_parquet(parquet_path, max_samples), output)
        print(f"Wrote {n} samples -> {output}")
        return output
    except Exception as exc:
        print(f"[modelscope/opus] parquet failed ({exc}), fallback to direct OPUS mirror")
        script = os.path.join(SCRIPT_DIR, "download_opus100_xen.py")
        cmd = [
            sys.executable,
            script,
            "--lang",
            "zh",
            "--output",
            output,
            "--cache_dir",
            os.path.join(cache_dir, "opus100"),
        ]
        if max_samples > 0:
            cmd.extend(["--max_samples", str(max_samples)])
        subprocess.run(cmd, check=True)
        return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="wmt",
        choices=("wmt", "nllb", "opus"),
        help="wmt=25M zh-en (large); nllb=mined; opus=~1M OPUS-100 en-zh",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/modelscope_zh_en.jsonl"),
    )
    parser.add_argument(
        "--cache_dir",
        default=os.path.join(ROOT, "data/parallel_cache/modelscope"),
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0=all available in downloaded files",
    )
    args = parser.parse_args()

    if args.source == "wmt":
        csv_path = download_wmt(args.cache_dir)
        n = export_jsonl(convert_wmt_csv(csv_path, args.max_samples), args.output)
    elif args.source == "nllb":
        gz_path = download_nllb_pair(args.cache_dir)
        n = export_jsonl(convert_nllb_gz(gz_path, args.max_samples), args.output)
    else:
        download_opus_enzh(args.output, args.cache_dir, args.max_samples)
        if not os.path.isfile(args.output):
            raise SystemExit(f"output missing: {args.output}")
        with open(args.output, "r", encoding="utf-8") as f:
            n = sum(1 for _ in f)

    print(f"Wrote {n} samples -> {args.output}")


if __name__ == "__main__":
    main()
