#!/usr/bin/env python3
"""Generate tgt for continue_en_half_50k via Instruct greedy continuation.

For each row in continue_en_half_50k.jsonl:
  prefix = src (no prompt)  ->  greedy generate  ->  tgt
  literal_tgt = original second-half tgt (kept for comparison)
  orig_tgt    = full original sentence (unchanged)
"""

import argparse
import json
import os
import time
from typing import Dict, List, Set

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_rows(path: str, limit: int = 0) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    return rows


def load_done_src(*paths: str) -> Set[str]:
    done: Set[str] = set()
    for output_path in paths:
        if not output_path or not os.path.isfile(output_path):
            continue
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                done.add(row["src"])
    return done


def shard_rows(rows: List[dict], shard_id: int, num_shards: int) -> List[dict]:
    if num_shards <= 1:
        return rows
    return [r for i, r in enumerate(rows) if i % num_shards == shard_id]


@torch.no_grad()
def greedy_continuations(model, tokenizer, prefixes: List[str], max_new_tokens: int, device: str) -> List[str]:
    enc = tokenizer(
        prefixes,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_len = input_ids.shape[1]

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    texts: List[str] = []
    for i in range(out.size(0)):
        # Must slice after full padded input length, not attention_mask sum.
        cont_ids = out[i, input_len:]
        texts.append(tokenizer.decode(cont_ids, skip_special_tokens=True))
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "data/encoder_only/continue_en_half_50k.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/continue_en_half_50k_instruct.jsonl"),
    )
    parser.add_argument(
        "--llm_path",
        default=os.path.join(ROOT, "model/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="", help="e.g. cuda:0")
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument(
        "--also_skip_from",
        default="",
        help="Extra jsonl whose src values are treated as done (resume aid).",
    )
    args = parser.parse_args()

    rows = load_rows(args.input, limit=args.limit)
    rows = shard_rows(rows, args.shard_id, args.num_shards)

    if args.num_shards > 1 and not args.output.endswith(f".shard{args.shard_id}.jsonl"):
        base, ext = os.path.splitext(args.output)
        args.output = f"{base}.shard{args.shard_id}{ext}"

    skip_paths = [args.output]
    if args.also_skip_from:
        skip_paths.append(args.also_skip_from)
    done: Set[str] = set()
    if args.resume:
        done = load_done_src(*skip_paths)
        rows = [r for r in rows if r["src"] not in done]

    print(f"LLM: {args.llm_path}")
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Shard: {args.shard_id}/{args.num_shards}")
    print(f"Rows to process: {len(rows)} (+ {len(done)} done)")
    print("prefix: src only (no prompt); tgt: raw greedy continuation")

    if not rows:
        print("Nothing to process.")
        return

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.llm_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device,
    )
    model.eval()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    mode = "a" if args.resume and os.path.isfile(args.output) else "w"
    written = 0
    t0 = time.time()

    with open(args.output, mode, encoding="utf-8") as fout:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prefixes = [r["src"] for r in batch]
            tgts = greedy_continuations(model, tokenizer, prefixes, args.max_new_tokens, device)
            for row, tgt in zip(batch, tgts):
                rec = {
                    "task": "continue_greedy",
                    "src_lang": row.get("src_lang", "eng_Latn"),
                    "src": row["src"],
                    "tgt": tgt,
                    "literal_tgt": row["tgt"],
                    "orig_tgt": row["orig_tgt"],
                    "label_source": "Meta-Llama-3-8B-Instruct_greedy_raw",
                    "max_new_tokens": args.max_new_tokens,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if (start // args.batch_size) % 50 == 0 or start + args.batch_size >= len(rows):
                elapsed = time.time() - t0
                n = start + len(batch)
                print(
                    f"  [{n}/{len(rows)}] written={written} ({n / max(elapsed, 1e-6):.1f} sent/s)",
                    flush=True,
                )

    print(f"Wrote {written} samples -> {args.output}")
    if written:
        with open(args.output, "r", encoding="utf-8") as f:
            print("Sample:")
            print(json.dumps(json.loads(f.readline()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
