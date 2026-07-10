#!/usr/bin/env python3
"""Train only mapping_enc2llm for multilingual x -> English generation.

This is a lighter Stage 1 warmup:

    x -> MT encoder -> mapping_enc2llm -> frozen LLM -> English en

Only mapping_enc2llm is trainable. The NLLB decoder, mapping_llm2dec, decoder CE,
and OT branches are disabled through llm_only=True.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling_xbridge import LlamaForCasualLMWithXBridge, XBridgeConfig  # noqa: E402


@dataclass
class TrainExample:
    src_lang: str
    src: str
    tgt: str
    prompt: str = ""


class XEnJsonlDataset(Dataset):
    def __init__(self, path: str):
        self.examples: List[TrainExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                src = row.get("src", row.get("source", row.get("input", "")))
                tgt = row.get("tgt", row.get("target", row.get("english", row.get("en", ""))))
                src_lang = row.get("src_lang", row.get("source_lang", row.get("lang", "")))
                prompt = row.get("prompt", row.get("instruction", ""))
                if not src or not tgt or not src_lang:
                    raise ValueError(f"Missing src/tgt/src_lang at line {line_no}: {row}")
                self.examples.append(TrainExample(src_lang=src_lang, src=src, tgt=tgt, prompt=prompt))
        if not self.examples:
            raise ValueError(f"No examples loaded from {path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]


def reset_mapping_weights(module: torch.nn.Module):
    for sub in module.modules():
        if isinstance(sub, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(sub.weight)
            if sub.bias is not None:
                torch.nn.init.zeros_(sub.bias)
        elif sub.__class__.__name__ == "LlamaRMSNorm" and hasattr(sub, "weight"):
            torch.nn.init.ones_(sub.weight)
    if hasattr(module, "end_boundary"):
        torch.nn.init.normal_(module.end_boundary, std=0.02)


def collate_x_en(batch: List[TrainExample], tokenizer_mt, tokenizer_llm, max_src_len: int, max_tgt_len: int, max_prompt_len: int):
    pad_llm = tokenizer_llm.pad_token_id
    if pad_llm is None:
        pad_llm = tokenizer_llm.eos_token_id or 0

    input_rows: List[List[int]] = []
    attention_rows: List[List[int]] = []
    aug_rows: List[List[int]] = []

    for ex in batch:
        tokenizer_mt.src_lang = ex.src_lang
        mt_ids = tokenizer_mt(
            ex.src,
            add_special_tokens=True,
            truncation=True,
            max_length=max_src_len,
            return_tensors=None,
        )["input_ids"]

        prompt_ids: List[int] = []
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
        aug = [1] * len(mt_ids) + [2] * len(prompt_ids) + [3] * len(tgt_ids)
        input_rows.append(seq)
        attention_rows.append([1] * len(seq))
        aug_rows.append(aug)

    max_len = max(len(x) for x in input_rows)
    padded_ids, padded_mask, padded_aug = [], [], []
    for ids, mask, aug in zip(input_rows, attention_rows, aug_rows):
        pad_len = max_len - len(ids)
        padded_ids.append([pad_llm] * pad_len + ids)
        padded_mask.append([0] * pad_len + mask)
        padded_aug.append([0] * pad_len + aug)

    ids_tensor = torch.tensor(padded_ids, dtype=torch.long)
    return {
        "input_ids": ids_tensor,
        "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
        "augmentation": torch.tensor(padded_aug, dtype=torch.long),
        "labels": ids_tensor.clone(),
    }


def build_model(args, len_tokenizer_llm: int, dtype: torch.dtype, device: str):
    config = XBridgeConfig(
        mt_path=args.mt_path,
        llm_path=args.llm_path,
        llm_only=True,
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
    model = LlamaForCasualLMWithXBridge(config, is_training=True, len_tokenizer_llm=len_tokenizer_llm)
    model = model.to(dtype=dtype)
    model.mapping_enc2llm = model.mapping_enc2llm.to(device)

    for p in model.parameters():
        p.requires_grad = False
    for p in model.mapping_enc2llm.parameters():
        p.requires_grad = True

    if args.resume_mapping:
        state = torch.load(args.resume_mapping, map_location="cpu")
        model.mapping_enc2llm.load_state_dict(state, strict=True)
        print(f"Loaded mapping_enc2llm from {args.resume_mapping}", flush=True)
    elif args.reinit_mapping:
        reset_mapping_weights(model.mapping_enc2llm)
        print("Re-initialized mapping_enc2llm from scratch.", flush=True)

    if hasattr(model.model_llm, "gradient_checkpointing_enable"):
        try:
            model.model_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            model.model_llm.gradient_checkpointing_enable()
    model.train()
    return model, config


@torch.no_grad()
def _flatten_cpu(tensors: Iterable[torch.Tensor]):
    tensors = list(tensors)
    if not tensors:
        return torch.empty(0, dtype=torch.float32)
    return torch.cat([t.detach().reshape(-1).float().cpu() for t in tensors])


@torch.no_grad()
def broadcast_params(params: List[torch.nn.Parameter], world_size: int):
    if world_size == 1:
        return
    flat = _flatten_cpu(params)
    dist.broadcast(flat, src=0)
    off = 0
    for p in params:
        n = p.numel()
        p.copy_(flat[off:off + n].view_as(p).to(p.device, p.dtype))
        off += n


@torch.no_grad()
def allreduce_grads(params: List[torch.nn.Parameter], world_size: int):
    if world_size == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = _flatten_cpu(grads)
    dist.all_reduce(flat)
    flat /= world_size
    off = 0
    for grad in grads:
        n = grad.numel()
        grad.copy_(flat[off:off + n].view_as(grad).to(grad.device, grad.dtype))
        off += n


def save_checkpoint(model, output_dir: str, config: XBridgeConfig, step, meta: dict):
    ckpt_dir = Path(output_dir) / f"checkpoint-{step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    config.save_pretrained(ckpt_dir)
    torch.save({k: v.detach().cpu() for k, v in model.mapping_enc2llm.state_dict().items()}, ckpt_dir / "mapping_enc2llm.pt")
    with (ckpt_dir / "train_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False, sort_keys=True)
    print(f"Saved checkpoint -> {ckpt_dir}", flush=True)


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train mapping_enc2llm on x->English data.")
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--output_dir", default="outputs/stage1_encoder_x_en")
    parser.add_argument("--mt_path", default="model/nllb-200-1.3B")
    parser.add_argument("--llm_path", default="model/Meta-Llama-3-8B")
    parser.add_argument("--per_device_batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=8,
                        help="DataLoader workers; tokenization runs here, off the GPU critical path")
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=256)
    parser.add_argument("--max_prompt_len", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reinit_mapping", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume_mapping", default="")
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    # Fast tokenizers already use Rust threads; silence the fork warning and avoid
    # oversubscription when DataLoader workers each hold a tokenizer copy.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    def log(*items):
        if is_main:
            print(*items, flush=True)

    set_seed(args.seed)
    if is_main:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.bf16 else torch.float16
    tokenizer_mt = AutoTokenizer.from_pretrained(args.mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(args.llm_path)
    if "llama-3" in args.llm_path.lower() or "llama3" in args.llm_path.lower():
        tokenizer_llm.pad_token_id = 128002
    elif tokenizer_llm.pad_token_id is None:
        tokenizer_llm.pad_token_id = tokenizer_llm.eos_token_id or 0
    tokenizer_llm.padding_side = "left"

    dataset = XEnJsonlDataset(args.train_file)
    # Pipeline tokenization across worker processes so the GPUs are not starved
    # waiting on per-batch NLLB/LLM tokenization done in the main process.
    loader_kwargs = dict(
        batch_size=args.per_device_batch_size,
        drop_last=True,
        collate_fn=lambda b: collate_x_en(b, tokenizer_mt, tokenizer_llm, args.max_src_len, args.max_tgt_len, args.max_prompt_len),
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, sampler=sampler, **loader_kwargs)
    else:
        sampler = None
        dataloader = DataLoader(dataset, shuffle=True, **loader_kwargs)

    model, config = build_model(args, len(tokenizer_llm), dtype, device)
    trainable = [p for p in model.mapping_enc2llm.parameters() if p.requires_grad]
    broadcast_params(trainable, world_size)

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(dataloader) / max(args.gradient_accumulation_steps, 1))
    total_steps = max(1, steps_per_epoch * args.num_epochs) if args.max_steps <= 0 else args.max_steps
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = args.per_device_batch_size * args.gradient_accumulation_steps * world_size
    log(f"world_size={world_size} | samples={len(dataset)} | eff_batch={eff_batch} | total_steps={total_steps} | warmup={warmup_steps}")
    log(f"trainable mapping_enc2llm params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    stop = False

    for epoch in range(args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(**batch, output_hidden_states=False)
                loss = out[0] / args.gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                allreduce_grads(trainable, world_size)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    avg = running_loss / args.logging_steps
                    log(f"epoch={epoch + 1} step={global_step} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0

                if is_main and args.save_steps > 0 and global_step % args.save_steps == 0:
                    save_checkpoint(model, args.output_dir, config, global_step, {
                        "stage": "encoder_x_en",
                        "trainable": "mapping_enc2llm",
                        "llm_only": True,
                        "global_step": global_step,
                        "epoch": epoch + 1,
                        "args": vars(args),
                    })

                if args.max_steps > 0 and global_step >= args.max_steps:
                    stop = True
                    break
        if stop:
            break
        if is_main:
            save_checkpoint(model, args.output_dir, config, f"epoch{epoch + 1}", {
                "stage": "encoder_x_en",
                "trainable": "mapping_enc2llm",
                "llm_only": True,
                "global_step": global_step,
                "epoch": epoch + 1,
                "args": vars(args),
            })

    if is_main:
        save_checkpoint(model, args.output_dir, config, "final" if not stop else f"stop{global_step}", {
            "stage": "encoder_x_en",
            "trainable": "mapping_enc2llm",
            "llm_only": True,
            "global_step": global_step,
            "args": vars(args),
        })
        log("Training finished." if not stop else f"Stopped at max_steps={args.max_steps}.")

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
