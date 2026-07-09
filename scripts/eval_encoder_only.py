#!/usr/bin/env python3
"""Evaluate encoder-only XBridge: NLLB encoder -> mapping_enc2llm -> LLM (x->en)."""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LANGS = ["en", "bn", "de", "es", "fr", "ja", "ru", "sw", "th", "zh"]
NON_EN = [l for l in LANGS if l != "en"]

PAPER_X_EN = {
    "bn-en": 37.09, "de-en": 45.75, "es-en": 32.00, "fr-en": 46.10,
    "ja-en": 27.63, "ru-en": 37.08, "sw-en": 44.73, "th-en": 30.61, "zh-en": 24.89,
    "x-en": 36.21,
}


def run_inference(args):
    cmd = [
        sys.executable,
        os.path.join(ROOT, "inference_xbridge_stage1.py"),
        "--mt_tokenizer_path", args.mt_path,
        "--llm_tokenizer_path", args.llm_path,
        "--base_model", args.base_model,
        "--batch_size", str(args.batch_size),
        "--testset_dir", args.testset_dir,
        "--output_dir", args.output_dir,
        "--trans_langs", ",".join(args.langs),
        "--max_new_tokens", str(args.max_new_tokens),
        "--mode", "encoder_only",
    ]
    if args.use_chat_template:
        cmd.append("--use_chat_template=True")
    print("[infer] encoder-only x->en (llm_only=True in stage1 script)")
    if args.use_chat_template:
        print("[infer] prompt: chat_inject Plan C (instruction + Enc(x) in user slot)")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=ROOT)


def score_x_en(output_dir, ref_dir, tokenize):
    from scripts.eval_flores import corpus_bleu, load_lines, FLORES_CODE

    rows = []
    for src in NON_EN:
        hyp_path = os.path.join(output_dir, f"encoder_only.{src}-en.en.llm")
        ref_path = os.path.join(ref_dir, "eng.devtest")
        if not os.path.isfile(hyp_path):
            raise FileNotFoundError(hyp_path)
        hyps = load_lines(hyp_path)
        refs = load_lines(ref_path)
        bleu = corpus_bleu(hyps, refs, tokenize=tokenize)
        direction = f"{src}-en"
        row = {
            "direction": direction,
            "bleu": round(bleu, 2),
            "n": len(hyps),
            "paper_bleu": PAPER_X_EN.get(direction),
            "bleu_diff": round(bleu - PAPER_X_EN[direction], 2) if direction in PAPER_X_EN else None,
        }
        rows.append(row)
    avg = sum(r["bleu"] for r in rows) / len(rows)
    summary = {
        "task": "encoder_only_x_to_en",
        "path": "NLLB_encoder -> mapping_enc2llm -> LLM",
        "x_en_bleu_avg": round(avg, 2),
        "paper_x_en_bleu": PAPER_X_EN["x-en"],
        "bleu_diff": round(avg - PAPER_X_EN["x-en"], 2),
        "base_model": None,
    }
    return summary, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=os.path.join(ROOT, "model/XBridge-base"))
    parser.add_argument("--mt_path", default=os.path.join(ROOT, "model/nllb-200-1.3B"))
    parser.add_argument("--llm_path", default=os.path.join(ROOT, "model/Meta-Llama-3-8B"))
    parser.add_argument("--testset_dir", default=os.path.join(ROOT, "data/flores101"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "outputs/encoder_only_eval"))
    parser.add_argument("--metrics_out", default=os.path.join(ROOT, "outputs/encoder_only_eval/metrics.json"))
    parser.add_argument("--langs", nargs="+", default=NON_EN)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--tokenize", default="flores200")
    parser.add_argument("--skip_infer", action="store_true")
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Wrap src with Instruct chat user template at inference (Plan A).",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    if not args.skip_infer:
        for src in args.langs:
            out = os.path.join(args.output_dir, f"encoder_only.{src}-en.en.llm")
            if os.path.isfile(out):
                os.remove(out)
        run_inference(args)

    sys.path.insert(0, ROOT)
    summary, rows = score_x_en(args.output_dir, args.testset_dir, args.tokenize)
    summary["base_model"] = args.base_model
    summary["use_chat_template"] = args.use_chat_template
    summary["interleave_enc_in_chat"] = args.use_chat_template
    payload = {"summary": summary, "directions": rows}
    os.makedirs(os.path.dirname(args.metrics_out) or ".", exist_ok=True)
    with open(args.metrics_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Encoder-only FLORES x->en ===")
    print(f"model: {args.base_model}")
    print(f"{'Direction':<8} {'BLEU':>6} {'Paper':>6} {'Diff':>6}")
    for r in sorted(rows, key=lambda x: x["direction"]):
        print(
            f"{r['direction']:<8} {r['bleu']:>6.2f} "
            f"{r.get('paper_bleu', ''):>6} {r.get('bleu_diff', ''):>6}"
        )
    print()
    print(
        f"X->En avg BLEU: {summary['x_en_bleu_avg']:.2f} "
        f"(paper {summary['paper_x_en_bleu']:.2f}, diff {summary['bleu_diff']:+.2f})"
    )
    print(f"Saved -> {args.metrics_out}")


if __name__ == "__main__":
    main()
