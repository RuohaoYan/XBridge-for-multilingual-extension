#!/usr/bin/env python3
"""Compare encoder-only inference: FP16 vs BF16 on FLORES x->en."""

import argparse
import gc
import os
import sys

import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

FLORES_MAP = {
    "zh": ("zho_simpl", "zho_Hans", "Chinese"),
    "de": ("deu", "deu_Latn", "German"),
    "ja": ("jpn", "jpn_Jpan", "Japanese"),
}


def load_model(base_model, dtype, device, max_new_tokens, tokenizer_llm):
    config = XBridgeConfig.from_pretrained(base_model)
    config.max_gen_len = max_new_tokens
    config.llm_only = True
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        len_tokenizer_llm=len(tokenizer_llm),
    )
    model.eval()
    return model


def generate_lines(model, lines, src_lang_name, nllb_code, tokenizer_mt, tokenizer_llm, device, batch_size):
    langs_map = {"English": "eng_Latn", src_lang_name: nllb_code}
    outputs = []
    for i in range(0, len(lines), batch_size):
        batch = lines[i : i + batch_size]
        ids_mt = []
        for line in batch:
            tokenizer_mt.src_lang = nllb_code
            ids_mt.append(
                tokenizer_mt(line, add_special_tokens=True, return_tensors=None)["input_ids"]
            )
        max_len = max(len(x) for x in ids_mt)
        pad = tokenizer_llm.pad_token_id
        input_ids = [[pad] * (max_len - len(s)) + s for s in ids_mt]
        mask = [[0] * (max_len - len(s)) + [1] * len(s) for s in ids_mt]
        aug = [[0] * (max_len - len(s)) + [1] * len(s) for s in ids_mt]
        with torch.no_grad():
            out = model(
                input_ids=torch.tensor(input_ids, device=device),
                attention_mask=torch.tensor(mask, device=device),
                augmentation=torch.tensor(aug, device=device),
                forced_decoder_start_token_id=tokenizer_mt.convert_tokens_to_ids(["eng_Latn"]),
            )
        decoded = tokenizer_llm.batch_decode(out[0], skip_special_tokens=True)
        outputs.extend(t.replace("\n", " ") for t in decoded)
    return outputs


def corpus_bleu(hyps, refs):
    import sacrebleu
    return sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score


def unload(model):
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_existing_hyps(output_dir, mode, lang, max_samples):
    path = os.path.join(output_dir, f"{mode}.{lang}-en.en.llm")
    with open(path, encoding="utf-8") as f:
        return [l.rstrip("\n") for l in f][:max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=os.path.join(ROOT, "model/XBridge-base"))
    parser.add_argument("--mt_path", default=os.path.join(ROOT, "model/nllb-200-1.3B"))
    parser.add_argument("--llm_path", default=os.path.join(ROOT, "model/Meta-Llama-3-8B"))
    parser.add_argument("--testset_dir", default=os.path.join(ROOT, "data/flores101"))
    parser.add_argument("--langs", default="zh,de,ja")
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--reuse_fp16_dir", default=os.path.join(ROOT, "outputs/encoder_only_eval"))
    parser.add_argument("--fp16_mode", default="encoder_only")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]

    tokenizer_mt = AutoTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    dtype_runs = [("bf16", torch.bfloat16)]
    all_outputs = {"fp16": {}, "bf16": {}}

    if args.reuse_fp16_dir and os.path.isdir(args.reuse_fp16_dir):
        print(f"\n=== Reusing FP16 outputs from {args.reuse_fp16_dir} ===")
        for lang in langs:
            flores_code, _, _ = FLORES_MAP[lang]
            ref_path = os.path.join(args.testset_dir, "eng.devtest")
            hyps = load_existing_hyps(args.reuse_fp16_dir, args.fp16_mode, lang, args.max_samples)
            with open(ref_path, encoding="utf-8") as f:
                ref_lines = [l.rstrip("\n") for l in f][: args.max_samples]
            all_outputs["fp16"][lang] = {"hyps": hyps, "refs": ref_lines}
            print(f"  {lang}->en: {len(hyps)} lines")
    else:
        dtype_runs = [("fp16", torch.float16)] + dtype_runs

    for dtype_name, dtype in dtype_runs:
        print(f"\n=== Loading {dtype_name} ===")
        model = load_model(args.base_model, dtype, device, args.max_new_tokens, tokenizer_llm)
        for lang in langs:
            flores_code, nllb_code, src_name = FLORES_MAP[lang]
            src_path = os.path.join(args.testset_dir, f"{flores_code}.devtest")
            ref_path = os.path.join(args.testset_dir, "eng.devtest")
            with open(src_path, encoding="utf-8") as f:
                src_lines = [l.rstrip("\n") for l in f][: args.max_samples]
            with open(ref_path, encoding="utf-8") as f:
                ref_lines = [l.rstrip("\n") for l in f][: args.max_samples]
            print(f"  generating {lang}->en ({len(src_lines)} samples) ...")
            hyps = generate_lines(
                model, src_lines, src_name, nllb_code,
                tokenizer_mt, tokenizer_llm, device, args.batch_size,
            )
            all_outputs[dtype_name][lang] = {"hyps": hyps, "refs": ref_lines}
        unload(model)

    print("\n=== FP16 vs BF16 comparison ===")
    print(f"{'Lang':<6} {'N':>5} {'Exact%':>8} {'FP16 BLEU':>10} {'BF16 BLEU':>10} {'BLEU diff':>10}")
    total_same = total_n = 0
    fp16_bleus, bf16_bleus = [], []

    for lang in langs:
        fp16_h = all_outputs["fp16"][lang]["hyps"]
        bf16_h = all_outputs["bf16"][lang]["hyps"]
        refs = all_outputs["fp16"][lang]["refs"]
        n = len(fp16_h)
        same = sum(a == b for a, b in zip(fp16_h, bf16_h))
        total_same += same
        total_n += n
        b_fp16 = corpus_bleu(fp16_h, refs)
        b_bf16 = corpus_bleu(bf16_h, refs)
        fp16_bleus.append(b_fp16)
        bf16_bleus.append(b_bf16)
        print(
            f"{lang:<6} {n:>5} {100*same/n:>7.1f}% {b_fp16:>10.2f} {b_bf16:>10.2f} {b_bf16-b_fp16:>+10.2f}"
        )
        if same < n:
            for i, (a, b) in enumerate(zip(fp16_h, bf16_h)):
                if a != b:
                    print(f"  first diff @{i}:")
                    print(f"    fp16: {a[:120]}")
                    print(f"    bf16: {b[:120]}")
                    break

    avg_fp16 = sum(fp16_bleus) / len(fp16_bleus)
    avg_bf16 = sum(bf16_bleus) / len(bf16_bleus)
    print()
    print(f"Overall exact match: {100*total_same/total_n:.1f}% ({total_same}/{total_n})")
    print(f"Avg BLEU  fp16={avg_fp16:.2f}  bf16={avg_bf16:.2f}  diff={avg_bf16-avg_fp16:+.2f}")


if __name__ == "__main__":
    main()
