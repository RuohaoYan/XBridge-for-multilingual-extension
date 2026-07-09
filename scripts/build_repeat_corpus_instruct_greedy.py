#!/usr/bin/env python3
"""Build repeat corpus: Instruct greedy causal labels with repeat prompt.

Causal prefix (no chat_template):
  Repeat the following sentence exactly: {src}\\n  ->  greedy generate  ->  tgt

tgt is saved as-is (no cleaning, no length filter).
"""

import argparse
import json
import os
import re
import time
from typing import Dict, List, Set, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REPEAT_PROMPT = "Repeat the following sentence exactly: {sentence}\n"


def normalize_sentence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def make_prompt(src: str) -> str:
    return REPEAT_PROMPT.format(sentence=src)


def collect_sentences(
    path: str,
    dedupe: bool = True,
    min_chars: int = 3,
    max_chars: int = 512,
    limit: int = 0,
) -> Tuple[List[str], Dict[str, int]]:
    seen: Set[str] = set()
    sentences: List[str] = []
    skipped = {"empty": 0, "short": 0, "long": 0, "dup": 0}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sent = normalize_sentence(row.get("tgt") or row.get("en") or row.get("src") or "")
            if not sent:
                skipped["empty"] += 1
                continue
            if len(sent) < min_chars:
                skipped["short"] += 1
                continue
            if len(sent) > max_chars:
                skipped["long"] += 1
                continue
            key = sent.casefold()
            if dedupe:
                if key in seen:
                    skipped["dup"] += 1
                    continue
                seen.add(key)
            sentences.append(sent)
            if limit and len(sentences) >= limit:
                break
    return sentences, skipped


def load_done_src(output_path: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.isfile(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            done.add(row["src"])
    return done


@torch.no_grad()
def greedy_continuations(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
) -> List[str]:
    enc = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=768,
        return_tensors="pt",
        add_special_tokens=True,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    input_lens = attention_mask.sum(dim=1)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    texts: List[str] = []
    for i in range(out.size(0)):
        cont_ids = out[i, input_lens[i] :]
        text = tokenizer.decode(cont_ids, skip_special_tokens=True)
        texts.append(text)
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=os.path.join(ROOT, "data/encoder_only/opus100_zh_en_100k.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(ROOT, "data/encoder_only/repeat_en_instruct_greedy.jsonl"),
    )
    parser.add_argument(
        "--llm_path",
        default=os.path.join(ROOT, "model/Meta-Llama-3-8B-Instruct"),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--min_chars", type=int, default=3)
    parser.add_argument("--max_chars", type=int, default=512)
    parser.add_argument("--no_dedupe", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    sentences, skipped = collect_sentences(
        args.input,
        dedupe=not args.no_dedupe,
        min_chars=args.min_chars,
        max_chars=args.max_chars,
        limit=args.limit,
    )

    done: Set[str] = set()
    if args.resume:
        done = load_done_src(args.output)
        sentences = [s for s in sentences if s not in done]

    print(f"LLM: {args.llm_path}")
    print(f"Prefix: {REPEAT_PROMPT!r}")
    print(f"Input sentences: {len(sentences)} (+ {len(done)} already done)")
    print("tgt: raw greedy continuation (no post-processing)")
    print(f"Skipped from source: {skipped}")

    if not sentences:
        print("Nothing to process.")
        return

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device} ...")
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
        for start in range(0, len(sentences), args.batch_size):
            batch = sentences[start : start + args.batch_size]
            prompts = [make_prompt(s) for s in batch]
            tgts = greedy_continuations(
                model, tokenizer, prompts, args.max_new_tokens, device
            )
            for src, prompt, tgt in zip(batch, prompts, tgts):
                rec = {
                    "task": "repeat_greedy",
                    "src_lang": "eng_Latn",
                    "src": src,
                    "prompt": prompt,
                    "tgt": tgt,
                    "label_source": "Meta-Llama-3-8B-Instruct_greedy_raw",
                    "max_new_tokens": args.max_new_tokens,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                written += 1

            if (start // args.batch_size) % 50 == 0 or start + args.batch_size >= len(sentences):
                elapsed = time.time() - t0
                done_n = start + len(batch)
                rate = done_n / max(elapsed, 1e-6)
                print(
                    f"  [{done_n}/{len(sentences)}] written={written} ({rate:.1f} sent/s)",
                    flush=True,
                )

    print(f"Output: {args.output}")
    print(f"Wrote {written} samples")
    if written:
        with open(args.output, "r", encoding="utf-8") as f:
            sample = json.loads(f.readline())
        print("Sample:")
        print(json.dumps(sample, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
