#!/usr/bin/env python3
"""Train encoder-only XBridge: NLLB encoder -> mapping_enc2llm -> frozen LLM."""

import json
import math
import os
import random
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

import fire
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from modeling_xbridge import LlamaForCasualLMWithXBridge, XBridgeConfig


@dataclass
class TrainExample:
    src_lang: str
    src: str
    tgt: str
    prompt: str = ""


class EncoderOnlyJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.examples: List[TrainExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                self.examples.append(
                    TrainExample(
                        src_lang=row["src_lang"],
                        src=row["src"],
                        tgt=row["tgt"],
                        prompt=row.get("prompt", ""),
                    )
                )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int) -> TrainExample:
        return self.examples[idx]


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reset_mapping_weights(module: nn.Module):
    for sub in module.modules():
        if isinstance(sub, nn.Linear):
            nn.init.xavier_uniform_(sub.weight)
            if sub.bias is not None:
                nn.init.zeros_(sub.bias)
        elif hasattr(sub, "weight") and sub.__class__.__name__ == "LlamaRMSNorm":
            nn.init.ones_(sub.weight)
    if hasattr(module, "end_boundary"):
        nn.init.normal_(module.end_boundary, std=0.02)


def collate_encoder_only(
    batch: List[TrainExample],
    tokenizer_mt,
    tokenizer_llm,
    max_src_len: int,
    max_tgt_len: int,
    max_prompt_len: int,
):
    pad_llm = tokenizer_llm.pad_token_id
    pad_mt = tokenizer_mt.pad_token_id

    input_ids_batch = []
    attention_batch = []
    augmentation_batch = []

    for ex in batch:
        tokenizer_mt.src_lang = ex.src_lang
        mt_ids = tokenizer_mt(
            ex.src,
            add_special_tokens=True,
            truncation=True,
            max_length=max_src_len,
            return_tensors=None,
        )["input_ids"]

        prompt_ids = []
        if ex.prompt:
            prompt_ids = tokenizer_llm(
                ex.prompt,
                add_special_tokens=False,
                truncation=True,
                max_length=max_prompt_len,
                return_tensors=None,
            )["input_ids"]

        tgt_ids = tokenizer_llm(
            ex.tgt,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tgt_len,
            return_tensors=None,
        )["input_ids"]
        if tokenizer_llm.eos_token_id is not None:
            tgt_ids = tgt_ids + [tokenizer_llm.eos_token_id]

        seq = mt_ids + prompt_ids + tgt_ids
        mt_len = len(mt_ids)
        prompt_len = len(prompt_ids)
        tgt_len = len(tgt_ids)

        aug = (
            [0] * (len(seq) - mt_len - prompt_len - tgt_len)
            + [1] * mt_len
            + [2] * prompt_len
            + [3] * tgt_len
        )
        # left-pad to batch max later; aug uses 0 for pad
        input_ids_batch.append(seq)
        attention_batch.append([1] * len(seq))
        augmentation_batch.append(aug)

    max_len = max(len(s) for s in input_ids_batch)
    padded_ids = []
    padded_mask = []
    padded_aug = []
    for ids, mask, aug in zip(input_ids_batch, attention_batch, augmentation_batch):
        pad_len = max_len - len(ids)
        padded_ids.append([pad_llm] * pad_len + ids)
        padded_mask.append([0] * pad_len + mask)
        padded_aug.append([0] * pad_len + aug)

    device = "cpu"
    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        "augmentation": torch.tensor(padded_aug, dtype=torch.long),
        "labels": torch.tensor(padded_ids, dtype=torch.long),
    }


def save_trainable_checkpoint(model, output_dir: str, config: XBridgeConfig, step: int):
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    config.save_pretrained(ckpt_dir)
    mapping_state = {k: v.cpu() for k, v in model.mapping_enc2llm.state_dict().items()}
    torch.save(mapping_state, os.path.join(ckpt_dir, "mapping_enc2llm.pt"))
    meta = {
        "step": step,
        "trainable": "mapping_enc2llm",
        "llm_only": config.llm_only,
    }
    with open(os.path.join(ckpt_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved trainable checkpoint -> {ckpt_dir}")


def collate_stage1(
    batch: List[TrainExample],
    tokenizer_mt,
    tokenizer_llm,
    max_src_len: int,
    max_tgt_len: int,
    max_prompt_len: int,
    y_lang: str = "zho_Hans",
    eng_lang: str = "eng_Latn",
    max_y_len: int = 128,
    max_mt_label_len: int = 128,
):
    """Trilingual (x, en, y) collate for paper Stage 1.

    LLM side is identical to collate_encoder_only: [src(aug1) | inst(aug2) | en(aug3)].
    Adds, for the decoder + OT losses (modeling_xbridge.forward:394-434):
      mt_labels          = en re-tokenized as NLLB source (eng_Latn) -> OT re-encode; pad 0
      decoder_labels     = y in NLLB target space [y_lang, y.., eos];              pad -100
      decoder_input_ids  = [eos] + decoder_labels[:-1] (pre-shifted, M2M100);       pad 0
    y defaults to the source sentence itself (y = x), so the decoder reconstructs the
    source language from the LLM's English-position hidden states.
    """
    pad_llm = tokenizer_llm.pad_token_id
    mt_eos = tokenizer_mt.eos_token_id  # 2

    input_ids_batch, attention_batch, augmentation_batch = [], [], []
    mt_label_rows, dec_in_rows, dec_lab_rows = [], [], []

    for ex in batch:
        # --- LLM side (source aug1 | prompt aug2 | english target aug3) ---
        tokenizer_mt.src_lang = ex.src_lang
        mt_ids = tokenizer_mt(ex.src, add_special_tokens=True, truncation=True,
                              max_length=max_src_len, return_tensors=None)["input_ids"]
        prompt_ids = []
        if ex.prompt:
            prompt_ids = tokenizer_llm(ex.prompt, add_special_tokens=False, truncation=True,
                                       max_length=max_prompt_len, return_tensors=None)["input_ids"]
        tgt_ids = tokenizer_llm(ex.tgt, add_special_tokens=False, truncation=True,
                                max_length=max_tgt_len, return_tensors=None)["input_ids"]
        if tokenizer_llm.eos_token_id is not None:
            tgt_ids = tgt_ids + [tokenizer_llm.eos_token_id]
        seq = mt_ids + prompt_ids + tgt_ids
        aug = [1] * len(mt_ids) + [2] * len(prompt_ids) + [3] * len(tgt_ids)
        input_ids_batch.append(seq)
        attention_batch.append([1] * len(seq))
        augmentation_batch.append(aug)

        # --- OT target: english re-encoded by NLLB (src_lang = eng_Latn) ---
        tokenizer_mt.src_lang = eng_lang
        mt_lab = tokenizer_mt(ex.tgt, add_special_tokens=True, truncation=True,
                              max_length=max_mt_label_len, return_tensors=None)["input_ids"]

        # --- decoder target y (= source text, in y_lang) : [y_lang, y.., eos] ---
        tokenizer_mt.src_lang = y_lang
        y_ids = tokenizer_mt(ex.src, add_special_tokens=True, truncation=True,
                             max_length=max_y_len, return_tensors=None)["input_ids"]
        dec_in = [mt_eos] + y_ids[:-1]      # pre-shifted: [eos, y_lang, y..]
        assert 0 not in mt_lab and 0 not in y_ids, "sentinel 0 leaked into NLLB content"
        mt_label_rows.append(mt_lab)
        dec_lab_rows.append(y_ids)
        dec_in_rows.append(dec_in)

    # LLM side: left-pad
    max_len = max(len(s) for s in input_ids_batch)
    padded_ids, padded_mask, padded_aug = [], [], []
    for ids, mask, aug in zip(input_ids_batch, attention_batch, augmentation_batch):
        p = max_len - len(ids)
        padded_ids.append([pad_llm] * p + ids)
        padded_mask.append([0] * p + mask)
        padded_aug.append([0] * p + aug)

    def rpad(rows, fill):
        m = max(len(r) for r in rows)
        return [r + [fill] * (m - len(r)) for r in rows]

    return {
        "input_ids": torch.tensor(padded_ids, dtype=torch.long),
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        "augmentation": torch.tensor(padded_aug, dtype=torch.long),
        "labels": torch.tensor(padded_ids, dtype=torch.long),
        "mt_labels": torch.tensor(rpad(mt_label_rows, 0), dtype=torch.long),          # OT, pad 0
        "decoder_input_ids": torch.tensor(rpad(dec_in_rows, 0), dtype=torch.long),    # pad 0
        "decoder_labels": torch.tensor(rpad(dec_lab_rows, -100), dtype=torch.long),   # pad -100
    }


def save_stage1_checkpoint(model, output_dir: str, config: XBridgeConfig, step, meta_extra: dict = None):
    """Persist all Stage-1 trainable components. mapping_enc2llm.pt keeps the exact
    format the zh->en eval / merge scripts expect."""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
    os.makedirs(ckpt_dir, exist_ok=True)
    config.save_pretrained(ckpt_dir)
    torch.save({k: v.cpu() for k, v in model.mapping_enc2llm.state_dict().items()},
               os.path.join(ckpt_dir, "mapping_enc2llm.pt"))
    torch.save({k: v.cpu() for k, v in model.mapping_llm2dec.state_dict().items()},
               os.path.join(ckpt_dir, "mapping_llm2dec.pt"))
    cross_attn = {k: v.cpu() for k, v in model.model_mt.state_dict().items()
                  if "decoder" in k and "encoder_attn" in k}
    torch.save(cross_attn, os.path.join(ckpt_dir, "decoder_cross_attn.pt"))
    meta = {"step": step, "trainable": "mapping_enc2llm+mapping_llm2dec+decoder.encoder_attn",
            "llm_only": config.llm_only}
    if meta_extra:
        meta.update(meta_extra)
    with open(os.path.join(ckpt_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved stage1 checkpoint -> {ckpt_dir}", flush=True)


def build_model(
    mt_path: str,
    llm_path: str,
    len_tokenizer_llm: int,
    llm_only: bool,
    reinit_mapping: bool,
    resume_mapping: str,
    dtype: torch.dtype,
    device: str,
):
    config = XBridgeConfig(
        mt_path=mt_path,
        llm_path=llm_path,
        llm_only=llm_only,
        train_device_map=device,
        freeze_enc=True,
        freeze_llm=True,
        freeze_dec=True,
        freeze_mapping_enc2llm=False,
        freeze_mapping_llm2dec=True,
        dec_lambda=0.0,
        ot_lambda=0.0,
        llm_bos_token_id=128000,
        llm_eos_token_id=128001,
        llm_pad_token_id=128002,
        mt_pad_token_id=1,
        mt_eos_token_id=2,
    )
    model = LlamaForCasualLMWithXBridge(
        config,
        is_training=True,
        len_tokenizer_llm=len_tokenizer_llm,
    )
    model = model.to(dtype=dtype)
    model.mapping_enc2llm = model.mapping_enc2llm.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.mapping_enc2llm.parameters():
        p.requires_grad = True

    if resume_mapping:
        state = torch.load(resume_mapping, map_location="cpu")
        model.mapping_enc2llm.load_state_dict(state, strict=True)
        print(f"Loaded mapping from {resume_mapping}")
    elif reinit_mapping:
        reset_mapping_weights(model.mapping_enc2llm)
        print("Re-initialized mapping_enc2llm from scratch.")

    model.model_llm.gradient_checkpointing_enable()
    model.train()
    return model, config


def main(
    train_file: str = "",
    output_dir: str = "outputs/encoder_only_train",
    mt_path: str = "",
    llm_path: str = "",
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    warmup_ratio: float = 0.03,
    max_src_len: int = 256,
    max_tgt_len: int = 256,
    max_prompt_len: int = 512,
    logging_steps: int = 10,
    save_steps: int = 500,
    max_steps: int = 0,
    seed: int = 42,
    reinit_mapping: bool = True,
    resume_mapping: str = "",
    bf16: bool = True,
):
    train_file = train_file or os.path.join(ROOT, "data/encoder_only/train.jsonl")
    mt_path = mt_path or os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = llm_path or os.path.join(ROOT, "model/Meta-Llama-3-8B")

    assert os.path.isfile(train_file), f"Missing train file: {train_file}"
    os.makedirs(output_dir, exist_ok=True)
    set_seed(seed)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    dataset = EncoderOnlyJsonlDataset(train_file)
    dataloader = DataLoader(
        dataset,
        batch_size=per_device_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_encoder_only(
            b,
            tokenizer_mt,
            tokenizer_llm,
            max_src_len,
            max_tgt_len,
            max_prompt_len,
        ),
        drop_last=True,
    )

    model, config = build_model(
        mt_path=mt_path,
        llm_path=llm_path,
        len_tokenizer_llm=len(tokenizer_llm),
        llm_only=True,
        reinit_mapping=reinit_mapping,
        resume_mapping=resume_mapping,
        dtype=dtype,
        device=device,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    total_steps = max(1, math.ceil(len(dataloader) / gradient_accumulation_steps) * num_epochs)
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"Samples: {len(dataset)} | steps/epoch: {len(dataloader)} | total optim steps: {total_steps}")
    print(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda" if device.startswith("cuda") else "cpu", dtype=dtype):
                out = model(**batch)
                loss = out[0] / gradient_accumulation_steps

            loss.backward()
            running_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    avg = running_loss / logging_steps
                    print(f"epoch={epoch+1} step={global_step} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if global_step % save_steps == 0:
                    save_trainable_checkpoint(model, output_dir, config, global_step)

                if max_steps > 0 and global_step >= max_steps:
                    save_trainable_checkpoint(model, output_dir, config, global_step)
                    print(f"Reached max_steps={max_steps}, stopping early.")
                    return

        save_trainable_checkpoint(model, output_dir, config, f"epoch{epoch+1}")

    save_trainable_checkpoint(model, output_dir, config, "final")
    print("Training finished.")


if __name__ == "__main__":
    fire.Fire(main)
