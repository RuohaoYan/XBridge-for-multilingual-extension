#!/usr/bin/env python3
"""Multi-GPU data-parallel training of mapping_enc2llm, gloo grad-sync.

NLLB encoder -> mapping_enc2llm -> frozen LLM. Only mapping_enc2llm (~21M) trains.

Why not DDP/NCCL: on this box (RTX 5090 / Blackwell + torch 2.7.1 bundled NCCL),
NCCL collectives >~1MB raise "illegal memory access". So each rank runs the full
frozen model independently on its own GPU (no cross-GPU compute), and only the
small mapping gradient is averaged across ranks via the gloo (CPU) backend, which
works fine here. Ranks start from identical mapping weights (broadcast once) and
apply identical averaged grads, so they stay in sync without per-step param sync.

Launch:
  torchrun --nproc_per_node=4 train_encoder_only_mp.py --train_file ... --output_dir ...
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
    collate_encoder_only,
    reset_mapping_weights,
    save_trainable_checkpoint,
)


def build_model(mt_path, llm_path, len_tokenizer_llm, reinit_mapping, resume_mapping, dtype, device):
    config = XBridgeConfig(
        mt_path=mt_path, llm_path=llm_path, llm_only=True, train_device_map=device,
        freeze_enc=True, freeze_llm=True, freeze_dec=True,
        freeze_mapping_enc2llm=False, freeze_mapping_llm2dec=True,
        dec_lambda=0.0, ot_lambda=0.0,
        llm_bos_token_id=128000, llm_eos_token_id=128001, llm_pad_token_id=128002,
        mt_pad_token_id=1, mt_eos_token_id=2,
    )
    model = LlamaForCasualLMWithXBridge(config, is_training=True, len_tokenizer_llm=len_tokenizer_llm)
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

    model.model_llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model, config


@torch.no_grad()
def broadcast_mapping(params, world_size):
    """Make every rank's mapping identical to rank 0 (one-time, via gloo/CPU)."""
    if world_size == 1:
        return
    for p in params:
        c = p.detach().cpu()
        dist.broadcast(c, src=0)
        p.copy_(c.to(p.device))


@torch.no_grad()
def allreduce_grads(params, world_size):
    """Average mapping grads across ranks via gloo (CPU)."""
    if world_size == 1:
        return
    for p in params:
        if p.grad is None:
            continue
        c = p.grad.detach().cpu()
        dist.all_reduce(c)          # sum across ranks
        c /= world_size             # -> mean
        p.grad.copy_(c.to(p.grad.device))


def main(
    train_file: str = "",
    output_dir: str = "outputs/encoder_only_mp",
    mt_path: str = "",
    llm_path: str = "",
    per_device_batch_size: int = 4,
    gradient_accumulation_steps: int = 2,
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
    instruction: str = "",
    bf16: bool = True,
):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    if world_size > 1:
        dist.init_process_group(backend="gloo")   # CPU backend for grad sync only
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"
    is_main = rank == 0

    def log(*a):
        if is_main:
            print(*a, flush=True)

    train_file = train_file or os.path.join(ROOT, "data/encoder_only/train.jsonl")
    mt_path = mt_path or os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = llm_path or os.path.join(ROOT, "model/Meta-Llama-3-8B")
    assert os.path.isfile(train_file), f"Missing train file: {train_file}"
    if is_main:
        os.makedirs(output_dir, exist_ok=True)

    # identical seed on every rank -> identical mapping init (kept in sync thereafter)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    dtype = torch.bfloat16 if bf16 else torch.float16

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    dataset = EncoderOnlyJsonlDataset(train_file)
    # Paper Stage 1 conditions the LLM on an English instruction prompt (Eq.2: p(z|x,inst)).
    # Inject it into every example's prompt slot -> sequence [BOS|Enc(x)|boundary|inst|tgt].
    if instruction:
        for ex in dataset.examples:
            ex.prompt = instruction
        tok_probe = AutoTokenizer.from_pretrained(llm_path)
        log(f"instruction applied to all {len(dataset)} examples: {instruction!r} "
            f"({len(tok_probe(instruction, add_special_tokens=False)['input_ids'])} llm tokens)")
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        dataloader = DataLoader(dataset, batch_size=per_device_batch_size, sampler=sampler, drop_last=True,
                                collate_fn=lambda b: collate_encoder_only(
                                    b, tokenizer_mt, tokenizer_llm, max_src_len, max_tgt_len, max_prompt_len))
    else:
        sampler = None
        dataloader = DataLoader(dataset, batch_size=per_device_batch_size, shuffle=True, drop_last=True,
                                collate_fn=lambda b: collate_encoder_only(
                                    b, tokenizer_mt, tokenizer_llm, max_src_len, max_tgt_len, max_prompt_len))

    model, config = build_model(
        mt_path=mt_path, llm_path=llm_path, len_tokenizer_llm=len(tokenizer_llm),
        reinit_mapping=reinit_mapping, resume_mapping=resume_mapping, dtype=dtype, device=device,
    )
    trainable = [p for p in model.mapping_enc2llm.parameters() if p.requires_grad]
    broadcast_mapping(trainable, world_size)   # guarantee identical start

    optimizer = torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)
    steps_per_epoch = math.ceil(len(dataloader) / gradient_accumulation_steps)
    total_steps = max(1, steps_per_epoch * num_epochs) if max_steps <= 0 else max_steps
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    eff_batch = per_device_batch_size * gradient_accumulation_steps * world_size
    log(f"world_size={world_size} | samples={len(dataset)} | per-rank batches/epoch={len(dataloader)}")
    log(f"effective batch={eff_batch} | total optim steps={total_steps} | warmup={warmup_steps}")
    log(f"Trainable params: {sum(p.numel() for p in trainable) / 1e6:.2f}M")

    global_step = 0
    running_loss = 0.0
    optimizer.zero_grad(set_to_none=True)
    stop = False

    for epoch in range(num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(**batch)
                loss = out[0] / gradient_accumulation_steps
            loss.backward()
            running_loss += loss.item()

            if (step + 1) % gradient_accumulation_steps == 0:
                allreduce_grads(trainable, world_size)      # sync mapping grads (gloo)
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
                    save_trainable_checkpoint(model, output_dir, config, global_step)
                if max_steps > 0 and global_step >= max_steps:
                    stop = True
                    break
        if not stop and is_main:
            save_trainable_checkpoint(model, output_dir, config, f"epoch{epoch+1}")
        if stop:
            break

    if is_main:
        save_trainable_checkpoint(model, output_dir, config, "final" if not stop else f"stop{global_step}")
        log("Training finished." if not stop else f"Stopped at max_steps={max_steps}.")
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
