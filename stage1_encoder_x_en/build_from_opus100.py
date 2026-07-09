#!/usr/bin/env python3
"""Build x->English JSONL for encoder Stage 1 warmup from mixed local sources.

Two source kinds are supported per language:

  * opus   - OPUS-100 (Helsinki-NLP/opus-100) parquet, English-centric. Each row is
             {"translation": {"en": "...", "<lang>": "..."}}. Stored under a single
             canonical dir whose name sorts the two codes alphabetically (en-zh, bn-en).
  * jsonl  - a pre-built JSONL with flexible fields (src/source/input, tgt/target/en...).
             Used e.g. for Swahili, whose NLLB x-en JSONL lives outside OPUS-100.

For every language the pipeline is identical: normalize -> filter -> dedupe ->
reservoir-sample up to the per-language cap. Output schema (see DATA_REQUIREMENTS.md):

    {"src_lang": "zho_Hans", "src": "...", "tgt": "English ...", "prompt": "Translate into English:"}

Writes one per-language file <out_dir>/<lang>_en.jsonl and one mixed shuffled file
<out_dir>/multilingual_x_en.jsonl. The mixed file is directly usable as TRAIN_FILE
and round-trips cleanly through build_x_en_data.py in DATA_MODE=json.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import pyarrow.parquet as pq


# lang -> source spec + NLLB code.
#   opus:  {"kind": "opus",  "dir": <opus-100 dir>, "key": <src key in translation struct>}
#   jsonl: {"kind": "jsonl", "path": <jsonl file>}
LANG_TABLE = {
    "zh": {"kind": "opus", "dir": "en-zh", "key": "zh", "nllb": "zho_Hans"},
    "bn": {"kind": "opus", "dir": "bn-en", "key": "bn", "nllb": "ben_Beng"},
    "th": {"kind": "opus", "dir": "en-th", "key": "th", "nllb": "tha_Thai"},
    "ja": {"kind": "opus", "dir": "en-ja", "key": "ja", "nllb": "jpn_Jpan"},
    "ru": {"kind": "opus", "dir": "en-ru", "key": "ru", "nllb": "rus_Cyrl"},
    "de": {"kind": "opus", "dir": "de-en", "key": "de", "nllb": "deu_Latn"},
    "fr": {"kind": "opus", "dir": "en-fr", "key": "fr", "nllb": "fra_Latn"},
    "es": {"kind": "opus", "dir": "en-es", "key": "es", "nllb": "spa_Latn"},
    # Swahili is not in the local OPUS-100 mirror; use the NLLB x-en JSONL.
    "sw": {"kind": "jsonl", "path": "data/encoder_only/opus100_sw_en_200k.jsonl", "nllb": "swh_Latn"},
}

SOURCE_FIELDS = ("src", "source", "input", "text", "x")
TARGET_FIELDS = ("tgt", "target", "english", "en", "translation", "z")

_ASCII_LETTER = re.compile(r"[A-Za-z]")


def normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def first_present(record: dict, fields) -> str:
    for field in fields:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value)
    return ""


def keep(src: str, tgt: str, min_chars: int, max_chars: int) -> bool:
    if len(src) < min_chars or len(tgt) < min_chars:
        return False
    if max_chars > 0 and (len(src) > max_chars or len(tgt) > max_chars):
        return False
    if src == tgt:  # untranslated / copied line
        return False
    if not _ASCII_LETTER.search(tgt):  # English target must contain latin letters
        return False
    return True


def iter_opus_pairs(spec: dict, opus_root: Path, batch_size: int = 50000):
    """Yield (src, tgt) from an OPUS-100 parquet train split."""
    key = spec["key"]
    files = sorted((opus_root / spec["dir"]).glob("train-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No train parquet in {opus_root / spec['dir']}")
    for path in files:
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=batch_size, columns=["translation"]):
            for pair in batch.column("translation").to_pylist():
                if not pair:
                    continue
                yield pair.get(key, "") or "", pair.get("en", "") or ""


def iter_jsonl_pairs(spec: dict):
    """Yield (src, tgt) from a JSONL file with flexible field names."""
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"JSONL source not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            yield first_present(row, SOURCE_FIELDS), first_present(row, TARGET_FIELDS)


def sample_language(lang: str, cap: int, opus_root: Path, min_chars: int,
                    max_chars: int, prompt: str, seed: int) -> list[dict]:
    """Reservoir-sample up to `cap` filtered, deduped examples for one language."""
    spec = LANG_TABLE[lang]
    nllb = spec["nllb"]
    if spec["kind"] == "opus":
        pairs = iter_opus_pairs(spec, opus_root)
    else:
        pairs = iter_jsonl_pairs(spec)

    rng = random.Random(seed)
    reservoir: list[dict] = []
    seen: set[tuple[str, str]] = set()
    n_seen = n_raw = n_kept = 0

    for raw_src, raw_tgt in pairs:
        n_raw += 1
        src = normalize_text(raw_src)
        tgt = normalize_text(raw_tgt)
        if not keep(src, tgt, min_chars, max_chars):
            continue
        key = (src, tgt)
        if key in seen:
            continue
        seen.add(key)
        n_kept += 1
        record = {"src_lang": nllb, "src": src, "tgt": tgt, "prompt": prompt}
        if len(reservoir) < cap:
            reservoir.append(record)
        else:
            j = rng.randint(0, n_seen)
            if j < cap:
                reservoir[j] = record
        n_seen += 1

    print(f"[{lang}] raw={n_raw} kept={n_kept} unique -> sampled={len(reservoir)} "
          f"(cap {cap}, kind={spec['kind']}, src_lang={nllb})")
    if len(reservoir) < cap:
        print(f"[{lang}] note: only {len(reservoir)} usable examples (< cap {cap})")
    return reservoir


def parse_langs(spec: str, default_cap: int) -> list[tuple[str, int]]:
    """Parse 'zh,bn,sw:200000' into [(lang, cap), ...]."""
    out = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" in item:
            lang, cap = item.split(":", 1)
            out.append((lang.strip(), int(cap)))
        else:
            out.append((item, default_cap))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--opus_root", default="data/Helsinki-NLP_opus-100")
    parser.add_argument("--out_dir", default="data/stage1_encoder_x_en")
    parser.add_argument("--langs", default="zh,bn,th,ja,ru,de,fr,es,sw:200000",
                        help="comma-separated 'lang[:cap]'; cap defaults to --per_lang")
    parser.add_argument("--per_lang", type=int, default=50000, help="default per-language cap")
    parser.add_argument("--prompt", default="Translate into English:")
    parser.add_argument("--min_chars", type=int, default=2)
    parser.add_argument("--max_chars", type=int, default=500, help="0 disables the upper bound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_name", default="multilingual_x_en.jsonl")
    args = parser.parse_args()

    lang_caps = parse_langs(args.langs, args.per_lang)
    unknown = [l for l, _ in lang_caps if l not in LANG_TABLE]
    if unknown:
        raise SystemExit(f"Unknown langs {unknown}. Known: {sorted(LANG_TABLE)}")

    opus_root = Path(args.opus_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixed: list[dict] = []
    for i, (lang, cap) in enumerate(lang_caps):
        records = sample_language(
            lang=lang, cap=cap, opus_root=opus_root,
            min_chars=args.min_chars, max_chars=args.max_chars,
            prompt=args.prompt, seed=args.seed + i,  # decorrelate per-language reservoirs
        )
        per_file = out_dir / f"{lang}_en.jsonl"
        with per_file.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[{lang}] wrote {len(records)} -> {per_file}")
        mixed.extend(records)

    random.Random(args.seed).shuffle(mixed)
    mixed_path = out_dir / args.mixed_name
    with mixed_path.open("w", encoding="utf-8") as f:
        for r in mixed:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[mixed] wrote {len(mixed)} -> {mixed_path}")


if __name__ == "__main__":
    main()
