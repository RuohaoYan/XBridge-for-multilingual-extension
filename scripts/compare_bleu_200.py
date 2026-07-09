#!/usr/bin/env python3
"""Compare FLORES-200-subset BLEU of XBridge-base vs XBridge-SFT."""
import os
import sacrebleu

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LANGS = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]
FLORES_CODE = {
    "en": "eng", "zh": "zho_simpl", "es": "spa", "fr": "fra", "th": "tha",
    "sw": "swh", "ja": "jpn", "bn": "ben", "de": "deu", "ru": "rus",
}
REF_DIR = os.path.join(ROOT, "data/flores101_200")
BASE_DIR = os.path.join(ROOT, "outputs/flores101_base_200")
SFT_DIR = os.path.join(ROOT, "outputs/flores101_sft_200_final")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f]


def bleu(hyp, ref):
    return round(sacrebleu.corpus_bleu(hyp, [ref], tokenize="flores200").score, 2)


def direction_file(src, tgt, kind):
    # kind: 'llm' for X->En, 'mt' for En->X
    if tgt == "en" and src != "en":
        return f"supervised.{src}-en.en.llm"
    if src == "en" and tgt != "en":
        return f"supervised.en-{tgt}.{tgt}.mt"
    return None


rows = []
for src in LANGS:
    for tgt in LANGS:
        if src == tgt:
            continue
        fname = direction_file(src, tgt, None)
        if fname is None:
            continue
        ref_path = os.path.join(REF_DIR, f"{FLORES_CODE[tgt]}.devtest")
        ref = load(ref_path)
        direction = f"{src}-{tgt}"
        base_path = os.path.join(BASE_DIR, fname)
        sft_path = os.path.join(SFT_DIR, fname)
        if not (os.path.exists(base_path) and os.path.exists(sft_path)):
            continue
        b = bleu(load(base_path), ref)
        s = bleu(load(sft_path), ref)
        rows.append((direction, b, s, round(s - b, 2)))

x_en = [r for r in rows if r[0].endswith("-en")]
en_x = [r for r in rows if r[0].startswith("en-")]


def avg(rs, i):
    return round(sum(r[i] for r in rs) / len(rs), 2)


print(f"{'Direction':<8} {'base':>7} {'SFT':>7} {'diff':>7}")
print("-" * 32)
for d, b, s, df in sorted(rows):
    print(f"{d:<8} {b:>7.2f} {s:>7.2f} {df:>+7.2f}")
print("-" * 32)
print(f"{'X->En':<8} {avg(x_en,1):>7.2f} {avg(x_en,2):>7.2f} {avg(x_en,3):>+7.2f}")
print(f"{'En->X':<8} {avg(en_x,1):>7.2f} {avg(en_x,2):>7.2f} {avg(en_x,3):>+7.2f}")
print(f"\n(n=200 sentences per direction, flores200 tokenization)")
