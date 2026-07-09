#!/usr/bin/env python3
"""Verify llm_only=True vs False produce identical LLM (.en.llm) outputs."""

import json
import os
import sys

import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

MGSM_SAMPLES = 3
FLORES_SAMPLES = 3
FLORES_LANGS = ("zh", "de", "ja")
MAX_NEW_TOKENS = 256


def load_model(base_model, llm_only, max_new_tokens, tokenizer_llm, device):
    config = XBridgeConfig.from_pretrained(base_model)
    config.max_gen_len = max_new_tokens
    config.llm_only = llm_only
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map=device,
        len_tokenizer_llm=len(tokenizer_llm),
    )
    model.eval()
    return model


def mgsm_batch(model, tokenizer_mt, tokenizer_llm, question, src_name, nllb_code, device):
    tokenizer_mt.src_lang = nllb_code
    ids_mt = [tokenizer_mt(question, add_special_tokens=True, return_tensors=None)["input_ids"]]
    prompt = (
        "Below is an instruction that describes a task. Write a response that "
        f"appropriately completes the request.\n\n### Instruction:\n{question}\n\n"
        "### Response: Let's think step by step."
    )
    ids_prompt = [tokenizer_llm(prompt, add_special_tokens=False, return_tensors=None)["input_ids"]]
    seqs = [a + b for a, b in zip(ids_mt, ids_prompt)]
    max_len = max(len(s) for s in seqs)
    pad = tokenizer_llm.pad_token_id
    input_ids = [[pad] * (max_len - len(s)) + s for s in seqs]
    mask = [[0] * (max_len - len(s)) + [1] * len(s) for s in seqs]
    aug = [
        [0] * (max_len - len(ids_mt[i]) - len(ids_prompt[i]))
        + [1] * len(ids_mt[i])
        + [2] * len(ids_prompt[i])
        for i in range(len(ids_mt))
    ]

    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(input_ids, device=device),
            attention_mask=torch.tensor(mask, device=device),
            augmentation=torch.tensor(aug, device=device),
            forced_decoder_start_token_id=tokenizer_mt.convert_tokens_to_ids(["eng_Latn"]),
        )
    return tokenizer_llm.decode(out[0][0], skip_special_tokens=True)


def flores_batch(model, tokenizer_mt, tokenizer_llm, line, nllb_code, device):
    tokenizer_mt.src_lang = nllb_code
    ids_mt = tokenizer_mt(line, add_special_tokens=True, return_tensors=None)["input_ids"]
    max_len = len(ids_mt)
    pad = tokenizer_llm.pad_token_id
    input_ids = [[pad] * (max_len - len(ids_mt)) + ids_mt]
    mask = [[0] * (max_len - len(ids_mt)) + [1] * len(ids_mt)]
    aug = [[0] * (max_len - len(ids_mt)) + [1] * len(ids_mt)]
    with torch.no_grad():
        out = model(
            input_ids=torch.tensor(input_ids, device=device),
            attention_mask=torch.tensor(mask, device=device),
            augmentation=torch.tensor(aug, device=device),
            forced_decoder_start_token_id=tokenizer_mt.convert_tokens_to_ids(["eng_Latn"]),
        )
    return tokenizer_llm.decode(out[0][0], skip_special_tokens=True)


def compare(name, text_a, text_b):
    same = text_a == text_b
    print(f"  [{name}] {'IDENTICAL' if same else 'DIFFERENT'}")
    if not same:
        print(f"    llm_only=False: {text_a[:120]}...")
        print(f"    llm_only=True : {text_b[:120]}...")
    return same


def run_with_mode(model, fn, *args, llm_only, **kwargs):
    model.config.llm_only = llm_only
    return fn(model, *args, **kwargs)


def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    sft = os.path.join(ROOT, "model/XBridge-SFT")
    base = os.path.join(ROOT, "model/XBridge-base")
    mt_path = os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = os.path.join(ROOT, "model/Meta-Llama-3-8B")

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    mgsm_map = {
        "zh": ("Chinese", "zho_Hans"),
        "de": ("German", "deu_Latn"),
    }
    flores_map = {"zh": "zho_simpl", "de": "deu", "ja": "jpn"}
    flores_nllb = {"zh": "zho_Hans", "de": "deu_Latn", "ja": "jpn_Jpan"}

    tasks = [
        ("MGSM", sft, "zh"),
        ("MGSM", sft, "de"),
        ("FLORES", base, "zh"),
        ("FLORES", base, "de"),
        ("FLORES", base, "ja"),
    ]

    all_ok = True
    current_ckpt = None
    model = None

    for task_type, ckpt, lang in tasks:
        print(f"\n=== {task_type} {lang} ({ckpt.split('/')[-1]}) ===")
        if ckpt != current_ckpt:
            if model is not None:
                del model
                torch.cuda.empty_cache()
            print(f"Loading {ckpt.split('/')[-1]} ...")
            model = load_model(ckpt, False, MAX_NEW_TOKENS, tokenizer_llm, device)
            current_ckpt = ckpt

        if task_type == "MGSM":
            data = json.load(open(os.path.join(ROOT, f"data/mgsm/mgsm_{lang}.json")))
            samples = data[:MGSM_SAMPLES]
            src_name, nllb = mgsm_map[lang]
            for i, row in enumerate(samples):
                q = row["question"]
                out_off = run_with_mode(
                    model, mgsm_batch, tokenizer_mt, tokenizer_llm, q, src_name, nllb, device, llm_only=False
                )
                out_on = run_with_mode(
                    model, mgsm_batch, tokenizer_mt, tokenizer_llm, q, src_name, nllb, device, llm_only=True
                )
                ok = compare(f"sample {i}", out_off, out_on)
                all_ok = all_ok and ok
        else:
            fpath = os.path.join(ROOT, f"data/flores101/{flores_map[lang]}.devtest")
            lines = [l.rstrip("\n") for l in open(fpath, encoding="utf-8")][:FLORES_SAMPLES]
            nllb = flores_nllb[lang]
            for i, line in enumerate(lines):
                out_off = run_with_mode(
                    model, flores_batch, tokenizer_mt, tokenizer_llm, line, nllb, device, llm_only=False
                )
                out_on = run_with_mode(
                    model, flores_batch, tokenizer_mt, tokenizer_llm, line, nllb, device, llm_only=True
                )
                ok = compare(f"sample {i}", out_off, out_on)
                all_ok = all_ok and ok

    print("\n=== Summary ===")
    if all_ok:
        print("All LLM outputs IDENTICAL between llm_only=False and llm_only=True.")
    else:
        print("Some outputs DIFFER — investigate non-determinism or path divergence.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
