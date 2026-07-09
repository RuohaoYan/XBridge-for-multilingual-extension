#!/usr/bin/env python3
"""Compute FLORES-101 BLEU/COMET for XBridge stage-1 outputs."""

import argparse
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANGS = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]
FLORES_CODE = {
    "en": "eng", "zh": "zho_simpl", "es": "spa", "fr": "fra",
    "th": "tha", "sw": "swh", "ja": "jpn", "bn": "ben",
    "de": "deu", "ru": "rus",
}

PAPER_BLEU = {
    "bn-en": 37.09, "en-bn": 28.42, "de-en": 45.75, "en-de": 35.45,
    "es-en": 32.00, "en-es": 29.59, "fr-en": 46.10, "en-fr": 49.38,
    "ja-en": 27.63, "en-ja": 20.12, "ru-en": 37.08, "en-ru": 30.57,
    "sw-en": 44.73, "en-sw": 34.68, "th-en": 30.61, "en-th": 17.09,
    "zh-en": 24.89, "en-zh": 23.11, "x-en": 36.21, "en-x": 29.82,
}

PAPER_COMET = {
    "bn-en": 88.53, "en-bn": 84.77, "de-en": 88.97, "en-de": 85.57,
    "es-en": 86.66, "en-es": 85.23, "fr-en": 88.92, "en-fr": 86.53,
    "ja-en": 87.01, "en-ja": 85.33, "ru-en": 86.23, "en-ru": 86.27,
    "sw-en": 85.63, "en-sw": 80.19, "th-en": 86.79, "en-th": 81.12,
    "zh-en": 84.64, "en-zh": 83.46, "x-en": 87.04, "en-x": 84.27,
}


def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def corpus_bleu(hypotheses, references, tokenize="13a"):
    import sacrebleu
    return sacrebleu.corpus_bleu(hypotheses, [references], tokenize=tokenize).score



def score_direction(src, tgt, output_dir, ref_dir, mode, tokenize):
    tgt_code = FLORES_CODE[tgt]

    if src == "en" and tgt != "en":
        hyp_path = os.path.join(output_dir, f"{mode}.en-{tgt}.{tgt}.mt")
        ref_path = os.path.join(ref_dir, f"{tgt_code}.devtest")
    elif tgt == "en" and src != "en":
        hyp_path = os.path.join(output_dir, f"{mode}.{src}-en.en.llm")
        ref_path = os.path.join(ref_dir, "eng.devtest")
    else:
        return None

    if not os.path.exists(hyp_path):
        raise FileNotFoundError(hyp_path)

    hypotheses = load_lines(hyp_path)
    references = load_lines(ref_path)
    if len(hypotheses) != len(references):
        raise ValueError(
            f"{src}->{tgt}: hyp {len(hypotheses)} vs ref {len(references)}"
        )

    bleu = corpus_bleu(hypotheses, references, tokenize=tokenize)
    result = {
        "direction": f"{src}-{tgt}",
        "bleu": round(bleu, 2),
        "n": len(hypotheses),
        "hyp": hyp_path,
        "ref": ref_path,
    }
    return result


def main(args):
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results = []
    comet_model = None
    if args.comet:
        from comet import download_model, load_from_checkpoint
        model_path = args.comet_model or download_model("Unbabel/wmt22-comet-da")
        comet_model = load_from_checkpoint(model_path)

    def maybe_comet(hypotheses, references, sources):
        if comet_model is None:
            return None
        data = [
            {"src": s, "mt": h, "ref": r}
            for s, h, r in zip(sources, hypotheses, references)
        ]
        gpus = 1 if args.gpus else 0
        scores = comet_model.predict(data, batch_size=8, gpus=gpus)
        if scores and scores[0] <= 1.0:
            return sum(scores) / len(scores) * 100
        return sum(scores) / len(scores)

    for src in LANGS:
        for tgt in LANGS:
            if src == tgt:
                continue
            row = score_direction(
                src, tgt, args.output_dir, args.ref_dir, args.mode, args.tokenize,
            )
            if row and comet_model is not None:
                src_code = FLORES_CODE[src]
                src_ref = load_lines(
                    os.path.join(args.ref_dir, f"{src_code}.devtest")
                )
                hypotheses = load_lines(row["hyp"])
                references = load_lines(row["ref"])
                row["comet"] = round(
                    maybe_comet(hypotheses, references, src_ref), 2
                )
            if row:
                key = row["direction"]
                if key in PAPER_BLEU:
                    row["paper_bleu"] = PAPER_BLEU[key]
                    row["bleu_diff"] = round(row["bleu"] - PAPER_BLEU[key], 2)
                if "comet" in row and key in PAPER_COMET:
                    row["paper_comet"] = PAPER_COMET[key]
                    row["comet_diff"] = round(row["comet"] - PAPER_COMET[key], 2)
                results.append(row)

    x_en = [r for r in results if r["direction"].endswith("-en")]
    en_x = [r for r in results if r["direction"].startswith("en-")]
    summary = {
        "x_en_bleu_avg": round(sum(r["bleu"] for r in x_en) / len(x_en), 2),
        "en_x_bleu_avg": round(sum(r["bleu"] for r in en_x) / len(en_x), 2),
        "paper_x_en_bleu": PAPER_BLEU["x-en"],
        "paper_en_x_bleu": PAPER_BLEU["en-x"],
    }
    if args.comet:
        summary["x_en_comet_avg"] = round(sum(r["comet"] for r in x_en) / len(x_en), 2)
        summary["en_x_comet_avg"] = round(sum(r["comet"] for r in en_x) / len(en_x), 2)
        summary["paper_x_en_comet"] = PAPER_COMET["x-en"]
        summary["paper_en_x_comet"] = PAPER_COMET["en-x"]

    payload = {"summary": summary, "directions": results}
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=== FLORES-101 BLEU (XBridge-base, LLaMA3-8B) ===")
    print(f"tokenize: {args.tokenize}")
    print(f"{'Direction':<8} {'BLEU':>6} {'Paper':>6} {'Diff':>6}", end="")
    if args.comet:
        print(f" {'COMET':>7} {'Paper':>7} {'Diff':>6}", end="")
    print()
    for r in sorted(results, key=lambda x: x["direction"]):
        diff = r.get("bleu_diff", "")
        line = f"{r['direction']:<8} {r['bleu']:>6.2f} {r.get('paper_bleu', ''):>6} {diff:>6}"
        if args.comet:
            line += f" {r['comet']:>7.2f} {r.get('paper_comet', ''):>7} {r.get('comet_diff', ''):>6}"
        print(line)
    print()
    print(
        f"X->En avg BLEU: {summary['x_en_bleu_avg']:.2f} "
        f"(paper {summary['paper_x_en_bleu']:.2f}, "
        f"diff {summary['x_en_bleu_avg'] - summary['paper_x_en_bleu']:+.2f})"
    )
    print(
        f"En->X avg BLEU: {summary['en_x_bleu_avg']:.2f} "
        f"(paper {summary['paper_en_x_bleu']:.2f}, "
        f"diff {summary['en_x_bleu_avg'] - summary['paper_en_x_bleu']:+.2f})"
    )
    if args.comet:
        print(
            f"X->En avg COMET: {summary['x_en_comet_avg']:.2f} "
            f"(paper {summary['paper_x_en_comet']:.2f})"
        )
        print(
            f"En->X avg COMET: {summary['en_x_comet_avg']:.2f} "
            f"(paper {summary['paper_en_x_comet']:.2f})"
        )
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        default=os.path.join(ROOT, "outputs/flores101"),
    )
    parser.add_argument(
        "--ref-dir",
        default=os.path.join(ROOT, "data/flores101"),
    )
    parser.add_argument("--mode", default="supervised")
    parser.add_argument("--tokenize", default="flores200")
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "outputs/flores101/metrics.json"),
    )
    parser.add_argument("--comet", action="store_true")
    parser.add_argument("--comet-model", default="", help="Local COMET checkpoint path")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    main(args)
