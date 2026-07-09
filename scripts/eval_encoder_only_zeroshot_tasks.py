#!/usr/bin/env python3
"""Evaluate encoder-only zero-shot task transfer (tasks NOT in Stage-1 translation training).

Default: MGSM zh — Chinese math reasoning with English Alpaca prompt.
Path: NLLB encoder -> mapping_enc2llm -> LLM (llm_only=True).
"""

import argparse
import json
import os
import re
import sys

import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge
from scripts.chat_prompt import (
    MGSM_INSTRUCTION,
    build_mgsm_inject_parts,
    pad_encoder_chat_inject_batch,
    supports_chat_template,
)

MGSM_PROMPT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{question}\n\n"
    "### Response: Let's think step by step."
)

# Tasks not covered by Stage-1 OPUS translation training
ZEROSHOT_TASKS = {
    "mgsm": {
        "name": "MGSM math reasoning",
        "trained_in_stage1": False,
        "trained_in_stage2": True,
        "description": "Chinese question + English CoT prompt -> English answer",
    },
}


def extract_last_num(text: str) -> float:
    text = re.sub(r"(\d),(\d)", r"\1\2", text)
    matches = re.findall(r"(\d+(?:\.\d+)?)", text)
    return float(matches[-1]) if matches else 0.0


def load_model(base_model, device, max_new_tokens, tokenizer_llm, interleave_enc_in_chat=False):
    config = XBridgeConfig.from_pretrained(base_model)
    config.max_gen_len = max_new_tokens
    config.llm_only = True
    config.interleave_enc_in_chat = interleave_enc_in_chat
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=device,
        len_tokenizer_llm=len(tokenizer_llm),
    )
    model.eval()
    return model


def predict_batch(
    model,
    questions,
    nllb_code,
    tokenizer_mt,
    tokenizer_llm,
    device,
    use_chat_template: bool = False,
    chat_prefix_ids=None,
    chat_suffix_ids=None,
):
    tokenizer_mt.src_lang = nllb_code
    ids_mt = [
        tokenizer_mt(q, add_special_tokens=True, return_tensors=None)["input_ids"]
        for q in questions
    ]
    if use_chat_template:
        prefix_list = [chat_prefix_ids for _ in questions]
        suffix_list = [chat_suffix_ids for _ in questions]
        input_ids, mask, aug = pad_encoder_chat_inject_batch(
            ids_mt, prefix_list, suffix_list, tokenizer_llm.pad_token_id
        )
    else:
        prompts = [MGSM_PROMPT.format(question=q) for q in questions]
        ids_prompt = [
            tokenizer_llm(p, add_special_tokens=False, return_tensors=None)["input_ids"]
            for p in prompts
        ]
        from scripts.chat_prompt import pad_encoder_prompt_batch

        input_ids, mask, aug = pad_encoder_prompt_batch(
            ids_mt, ids_prompt, tokenizer_llm.pad_token_id
        )
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(input_ids, device=device),
            attention_mask=torch.tensor(mask, device=device),
            augmentation=torch.tensor(aug, device=device),
        )
    return tokenizer_llm.batch_decode(out[0], skip_special_tokens=True)


def eval_mgsm(
    model,
    lang,
    mgsm_dir,
    nllb_code,
    tokenizer_mt,
    tokenizer_llm,
    device,
    batch_size,
    use_chat_template: bool = False,
    chat_prefix_ids=None,
    chat_suffix_ids=None,
):
    path = os.path.join(mgsm_dir, f"mgsm_{lang}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    questions = [d["question"] for d in data]
    answers = [extract_last_num(str(d["answer"])) for d in data]
    preds_text = []
    for i in range(0, len(questions), batch_size):
        batch_q = questions[i : i + batch_size]
        preds_text.extend(
            predict_batch(
                model,
                batch_q,
                nllb_code,
                tokenizer_mt,
                tokenizer_llm,
                device,
                use_chat_template=use_chat_template,
                chat_prefix_ids=chat_prefix_ids,
                chat_suffix_ids=chat_suffix_ids,
            )
        )
    preds = [extract_last_num(t) for t in preds_text]
    hits = sum(1 for p, g in zip(preds, answers) if p == g)
    acc = round(100.0 * hits / len(answers), 2)
    return {
        "lang": lang,
        "n": len(answers),
        "accuracy": acc,
        "hits": hits,
        "predictions": preds_text,
    }


MGSM_LANGS = {
    "zh": "zho_Hans",
    "en": "eng_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "sw": "swh_Latn",
    "bn": "ben_Beng",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", default=os.path.join(ROOT, "model/XBridge-base"))
    parser.add_argument("--mt_path", default=os.path.join(ROOT, "model/nllb-200-1.3B"))
    parser.add_argument("--llm_path", default=os.path.join(ROOT, "model/Meta-Llama-3-8B"))
    parser.add_argument("--mgsm_dir", default=os.path.join(ROOT, "data/mgsm"))
    parser.add_argument("--output_dir", default=os.path.join(ROOT, "outputs/encoder_only_zeroshot"))
    parser.add_argument("--langs", default="zh", help="comma-separated MGSM langs")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--compare_sft", default=os.path.join(ROOT, "outputs/mgsm_g3/accuracy"))
    parser.add_argument(
        "--use_chat_template",
        action="store_true",
        help="Use tokenizer.apply_chat_template for Instruct models (Plan A inference).",
    )
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer_mt = AutoTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    print("=== Encoder-only zero-shot task eval ===")
    print(f"model: {args.base_model}")
    print(f"path:  NLLB encoder -> mapping_enc2llm -> LLM (llm_only=True)")
    print(f"task:  MGSM (NOT in Stage-1 translation training)")
    print(f"langs: {langs}")
    print(f"prompt: {'chat_inject (Plan C)' if args.use_chat_template else 'alpaca'}")
    if args.use_chat_template and not supports_chat_template(tokenizer_llm):
        raise ValueError(f"--use_chat_template requires a chat tokenizer: {args.llm_path}")
    chat_prefix_ids = chat_suffix_ids = None
    if args.use_chat_template:
        chat_prefix_ids, chat_suffix_ids = build_mgsm_inject_parts(tokenizer_llm)
    print()

    model = load_model(
        args.base_model,
        device,
        args.max_new_tokens,
        tokenizer_llm,
        interleave_enc_in_chat=args.use_chat_template,
    )

    results = []
    for lang in langs:
        print(f"Evaluating MGSM {lang} ...")
        row = eval_mgsm(
            model,
            lang,
            args.mgsm_dir,
            MGSM_LANGS[lang],
            tokenizer_mt,
            tokenizer_llm,
            device,
            args.batch_size,
            use_chat_template=args.use_chat_template,
            chat_prefix_ids=chat_prefix_ids,
            chat_suffix_ids=chat_suffix_ids,
        )
        results.append(row)
        out_path = os.path.join(args.output_dir, f"mgsm_{lang}.en.llm")
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(t.replace("\n", " ") for t in row.pop("predictions")))
        print(f"  {lang}: accuracy={row['accuracy']}% ({row['hits']}/{row['n']})")

    sft_ref = {}
    for pattern_dir in [os.path.join(ROOT, "outputs", d) for d in os.listdir(os.path.join(ROOT, "outputs")) if d.startswith("mgsm")]:
        acc_path = os.path.join(pattern_dir, "accuracy")
        if not os.path.isfile(acc_path):
            continue
        with open(acc_path, encoding="utf-8") as f:
            for line in f:
                m = re.search(r"gsm_8k_(\w+)\.en\.llm:\s*([\d.]+)", line)
                if m:
                    sft_ref[m.group(1)] = float(m.group(2))

    payload = {
        "eval_type": "encoder_only_zeroshot_task",
        "base_model": args.base_model,
        "task": ZEROSHOT_TASKS["mgsm"],
        "prompt_template": (
            f"chat_inject Plan C ({MGSM_INSTRUCTION} + Enc(question))"
            if args.use_chat_template
            else "Alpaca + CoT (English)"
        ),
        "use_chat_template": args.use_chat_template,
        "interleave_enc_in_chat": args.use_chat_template,
        "note": "Stage-1 trains translation only; MGSM requires semantic+prompt transfer without task-specific fine-tuning.",
        "results": results,
        "xbridge_sft_reference": sft_ref,
    }
    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("\n=== Summary (zero-shot task, encoder-only) ===")
    print(f"{'Lang':<6} {'Acc':>7} {'SFT ref':>8} {'Gap':>7}")
    for r in results:
        ref = sft_ref.get(r["lang"])
        gap = f"{r['accuracy'] - ref:+.1f}" if ref is not None else "n/a"
        ref_s = f"{ref:.1f}" if ref is not None else "n/a"
        print(f"{r['lang']:<6} {r['accuracy']:>6.1f}% {ref_s:>7}% {gap:>7}")
    print(f"\nSaved -> {metrics_path}")


if __name__ == "__main__":
    main()
