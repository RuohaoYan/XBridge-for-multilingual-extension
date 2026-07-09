#!/usr/bin/env python3
"""Inference for mapping_enc2llm-only x->English checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling_xbridge import LlamaForCasualLMWithXBridge, XBridgeConfig  # noqa: E402


def read_inputs(path: str) -> List[str]:
    p = Path(path)
    if p.suffix.lower() == ".jsonl":
        out = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                text = row.get("src", row.get("source", row.get("input", row.get("text", ""))))
                if text:
                    out.append(str(text))
        return out
    with p.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def build_batch(texts, src_lang, prompt, tokenizer_mt, tokenizer_llm):
    pad_llm = tokenizer_llm.pad_token_id if tokenizer_llm.pad_token_id is not None else 0
    rows, aug_rows = [], []
    for text in texts:
        tokenizer_mt.src_lang = src_lang
        mt_ids = tokenizer_mt(text, add_special_tokens=True, truncation=False, return_tensors=None)["input_ids"]
        prompt_ids = tokenizer_llm(prompt, add_special_tokens=False, truncation=False, return_tensors=None)["input_ids"] if prompt else []
        rows.append(mt_ids + prompt_ids)
        aug_rows.append([1] * len(mt_ids) + [2] * len(prompt_ids))
    max_len = max(len(x) for x in rows)
    input_ids, attention_mask, augmentation = [], [], []
    for ids, aug in zip(rows, aug_rows):
        pad_len = max_len - len(ids)
        input_ids.append([pad_llm] * pad_len + ids)
        attention_mask.append([0] * pad_len + [1] * len(ids))
        augmentation.append([0] * pad_len + aug)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "augmentation": torch.tensor(augmentation, dtype=torch.long),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--mt_path", default="model/nllb-200-1.3B")
    parser.add_argument("--llm_path", default="model/Meta-Llama-3-8B")
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--src_lang", default="zho_Hans")
    parser.add_argument("--prompt", default="Translate into English:")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    tokenizer_mt = AutoTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path)
    if "llama-3" in args.llm_path.lower() or "llama3" in args.llm_path.lower():
        tokenizer_llm.pad_token_id = 128002
    elif tokenizer_llm.pad_token_id is None:
        tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id or 0
    tokenizer_llm.padding_side = "left"

    config = XBridgeConfig.from_pretrained(args.base_model)
    config.mt_path = args.mt_path
    config.llm_path = args.llm_path
    config.llm_only = True
    config.max_gen_len = args.max_new_tokens
    config.freeze_enc = True
    config.freeze_llm = True
    config.freeze_dec = True
    config.freeze_mapping_enc2llm = False
    config.freeze_mapping_llm2dec = True
    config.dec_lambda = 0.0
    config.ot_lambda = 0.0

    model = LlamaForCasualLMWithXBridge(config, is_training=True, len_tokenizer_llm=len(tokenizer_llm))
    mapping_path = Path(args.base_model) / "mapping_enc2llm.pt"
    model.mapping_enc2llm.load_state_dict(torch.load(mapping_path, map_location="cpu"), strict=True)
    model.eval()
    device = next(model.mapping_enc2llm.parameters()).device

    texts = read_inputs(args.input_file)
    outputs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = build_batch(texts[i:i + args.batch_size], args.src_lang, args.prompt, tokenizer_mt, tokenizer_llm)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs.extend(tokenizer_llm.batch_decode(model(**batch)[0], skip_special_tokens=True))

    Path(args.output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_file, "w", encoding="utf-8") as f:
        for output in outputs:
            f.write(output.replace("\n", " ").strip() + "\n")
    print(f"Wrote {len(outputs)} outputs to {args.output_file}")


if __name__ == "__main__":
    main()
