#!/usr/bin/env python3
"""Paper Stage 1 (Cross-Model Mapping) training, 4-GPU gloo grad-sync.

Trilingual (x, en, y) joint training of mapping_enc2llm + mapping_llm2dec +
NLLB decoder cross-attention, with L = 1.0*L_CE_LLM + 1.0*L_CE_Dec + 6.0*L_OT
(the three losses are already implemented in modeling_xbridge.forward:394-434 and
enabled by llm_only=False). Only ~300M params train; the frozen 8B LLM + NLLB
encoder stay on each GPU (no cross-GPU compute). Trainable grads are averaged
across ranks via gloo (NCCL is broken on this box; see train_encoder_only_mp.py).

Launch:
  torchrun --nproc_per_node=4 train_stage1_mp.py --train_file ... --instruction "..."
"""
import math
import os
import random
import sys

import fire
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from modeling_xbridge import LlamaForCasualLMWithXBridge, XBridgeConfig  # noqa: E402
from train_encoder_only import (  # noqa: E402
    EncoderOnlyJsonlDataset,
    collate_stage1,
    reset_mapping_weights,
    save_stage1_checkpoint,
)


def build_model(mt_path, llm_path, len_tokenizer_llm, reinit_mapping, resume_mapping, dtype, device):
    config = XBridgeConfig(
        mt_path=mt_path, llm_path=llm_path, llm_only=False, train_device_map=device,
        freeze_enc=True, freeze_llm=True, freeze_dec=True,          # freeze_dec auto-unfreezes decoder.encoder_attn
        freeze_mapping_enc2llm=False, freeze_mapping_llm2dec=False,
        dec_lambda=1.0, ot_lambda=6.0,
        llm_bos_token_id=128000, llm_eos_token_id=128001, llm_pad_token_id=128002,
        mt_pad_token_id=1, mt_eos_token_id=2,
    )
    model = LlamaForCasualLMWithXBridge(config, is_training=True, len_tokenizer_llm=len_tokenizer_llm)
    model = model.to(dtype=dtype)
    model.mapping_enc2llm = model.mapping_enc2llm.to(device)
    model.mapping_llm2dec = model.mapping_llm2dec.to(device)

    # NO blanket freeze: rely on __init__ freeze flags (enc2llm + llm2dec + decoder.encoder_attn trainable).
    if resume_mapping:
        model.mapping_enc2llm.load_state_dict(torch.load(resume_mapping, map_location="cpu"), strict=True)
        print(f"Loaded mapping_enc2llm from {resume_mapping}")
    elif reinit_mapping:
        reset_mapping_weights(model.mapping_enc2llm)
        reset_mapping_weights(model.mapping_llm2dec)
        print("Re-initialized mapping_enc2llm + mapping_llm2dec from scratch.")

    # Paper trains cross-attn only, not the output vocab projection: keep lm_head frozen.
    if hasattr(model.model_mt, "lm_head"):
        for p in model.model_mt.lm_head.parameters():
            p.requires_grad = False

    model.model_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, config


def trainable_groups(model):
    g = {"enc2llm": [], "llm2dec": [], "cross_attn": [], "other": []}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "mapping_enc2llm" in n:
            g["enc2llm"].append(p)
        elif "mapping_llm2dec" in n:
            g["llm2dec"].append(p)
        elif "encoder_attn" in n:
            g["cross_attn"].append(p)
        else:
            g["other"].append(p)
    return g


@torch.no_grad()
def _flatten_cpu(tensors):
    return torch.cat([t.detach().reshape(-1).float().cpu() for t in tensors])


@torch.no_grad()
def broadcast_params(params, world_size):
    """One flat gloo broadcast of all trainable weights from rank 0."""
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
def allreduce_grads(params, world_size):
    """One flat gloo all-reduce (mean) of all trainable grads."""
    if world_size == 1:
        return
    grads = [p.grad for p in params if p.grad is not None]
    flat = _flatten_cpu(grads)
    dist.all_reduce(flat)
    flat /= world_size
    off = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[off:off + n].view_as(g).to(g.device, g.dtype))
        off += n


def main(
    train_file: str = "",
    output_dir: str = "outputs/stage1_mp",
    mt_path: str = "",
    llm_path: str = "",
    per_device_batch_size: int = 2,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    warmup_ratio: float = 0.03,
    max_src_len: int = 256,
    max_tgt_len: int = 256,
    max_prompt_len: int = 512,
    max_y_len: int = 128,
    max_mt_label_len: int = 128,
    y_lang: str = "zho_Hans",
    logging_steps: int = 10,
    save_steps: int = 500,
    max_steps: int = 0,
    seed: int = 42,
    reinit_mapping: bool = True,
    resume_mapping: str = "",
    instruction: str = "",
    bf16: bool = True,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="gloo")
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    train_file = train_file or os.path.join(ROOT, "data/encoder_only/opus100_zh_en_200k.jsonl")
    mt_path = mt_path or os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = llm_path or os.path.join(ROOT, "model/Meta-Llama-3-8B")
    assert os.path.isfile(train_file), f"Missing train file: {train_file}"
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    dataset = EncoderOnlyJsonlDataset(train_file)
    if instruction:
        for ex in dataset.examples:
            ex.prompt = instruction
        log(f"instruction applied to all {len(dataset)} examples: {instruction!r}")

    def collate(b):
        return collate_stage1(b, tokenizer_mt, tokenizer_llm, max_src_len, max_tgt_len,
                              max_prompt_len, y_lang=y_lang, max_y_len=max_y_len,
                              max_mt_label_len=max_mt_label_len)

    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=per_device_batch_size, sampler=sampler,
                                drop_last=True, collate_fn=collate)
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=per_device_batch_size, shuffle=True,
                                drop_last=True, collate_fn=collate)

    model, config = build_model(
        mt_path=mt_path, llm_path=llm_path, len_tokenizer_llm=len(tokenizer_llm),
        reinit_mapping=reinit_mapping, resume_mapping=resume_mapping, dtype=dtype, device=device,
    )
    groups = trainable_groups(model)
    trainable = [p for g in groups.values() for p in g]
    assert model.model_mt.lm_head.weight.requires_grad is False, "lm_head must stay frozen"
    broadcast_params(trainable, world_size)  # identical start on all ranks

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * num_epochs) if max_steps <= 0 else max_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    log(f"world_size={world_size} | samples={len(dataset)} | eff_batch={eff_batch} | "
        f"total optim steps={total_steps} | warmup={warmup_steps}")
    log("trainable(M): " + ", ".join(f"{k}={sum(p.numel() for p in v)/1e6:.1f}"
                                      for k, v in groups.items() if v))

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    stop = False
    diagnosed = False

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype):
                # decoder branch reads llm penultimate hidden states (modeling:402) -> need this
                out = model(**batch, output_hidden_states=True)
                loss = out[0] / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if not diagnosed and is_main:  # one-time: prove all 3 loss branches reach their params
                diagnosed = True
                gn = {k: (sum(p.grad.detach().float().norm().item() ** 2 for p in v if p.grad is not None) ** 0.5)
                      for k, v in groups.items() if v}
                log(f"[diag] first-step grad norms by group: {gn} | combined_loss={out[0].item():.4f}")

            if (step + 1) % gradient_accumulation_steps == 0:
                allreduce_grads(trainable, world_size)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % logging_steps == 0:
                    log(f"epoch={epoch+1} step={global_step} loss={running_loss/logging_steps:.4f} "
                        f"lr={scheduler.get_last_lr()[0]:.2e}")
                    running_loss = 0.0
                if is_main and global_step % save_steps == 0:
                    save_stage1_checkpoint(model, output_dir, config, global_step,
                                           {"instruction": instruction, "y_lang": y_lang})
                if max_steps > 0 and global_step >= max_steps:
                    stop = True
                    break
        if not stop and is_main:
            save_stage1_checkpoint(model, output_dir, config, f"epoch{epoch+1}",
                                   {"instruction": instruction, "y_lang": y_lang})
        if stop:
            break

    if is_main:
        save_stage1_checkpoint(model, output_dir, config, "final" if not stop else f"stop{global_step}",
                               {"instruction": instruction, "y_lang": y_lang})
        log("Training finished." if not stop else f"Stopped at max_steps={max_steps}.")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
