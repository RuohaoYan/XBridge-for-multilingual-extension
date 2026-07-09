#!/usr/bin/env python3
"""把 continue_en_half_50k_instruct.jsonl 的 tgt 按 whitespace 词数截到各自 literal_tgt 的长度。

目的：消除上一轮实验里 instruct tgt（~52 词）远长于 base 字面 tgt（~8 词）的长度混淆。
截断后 instruct 的 tgt 与 base 的 tgt（= literal_tgt）逐样本词数相等，只剩"贪婪内容 vs gold 内容"
一个变量。同时把 base 文件子采样到两文件公共 orig_tgt 子集，使两组样本一一对应。

输出：
  continue_en_half_50k_instruct_truncated.jsonl  （tgt 截断后的 instruct，N 条）
  continue_en_half_50k_common.jsonl             （base 公共子集，N 条，与上者 orig_tgt 一一对应）
"""
import argparse
import json
import os
import statistics as st
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def word_split(s: str):
    return s.split()


def truncate_to_words(text: str, n_words: int) -> str:
    """按空格截到前 n_words 个词，保留原分隔风格（用空格 join）。"""
    words = text.split()
    if len(words) <= n_words:
        return text
    return " ".join(words[:n_words])


def summ(x):
    if not x:
        return "n=0"
    xs = sorted(x)
    return (f"n={len(x)} mean={st.mean(x):.2f} median={st.median(xs):.0f} "
            f"min={min(xs)} max={max(xs)} p95={xs[int(0.95 * len(xs))]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instruct", default=os.path.join(
        ROOT, "data/encoder_only/continue_en_half_50k_instruct.jsonl"))
    ap.add_argument("--base", default=os.path.join(
        ROOT, "data/encoder_only/continue_en_half_50k.jsonl"))
    ap.add_argument("--out_instruct", default=os.path.join(
        ROOT, "data/encoder_only/continue_en_half_50k_instruct_truncated.jsonl"))
    ap.add_argument("--out_base", default=os.path.join(
        ROOT, "data/encoder_only/continue_en_half_50k_common.jsonl"))
    ap.add_argument("--key", default="orig_tgt",
                    choices=["orig_tgt", "src"],
                    help="用于匹配 base 与 instruct 的字段")
    args = ap.parse_args()

    # ---- 1. 读 instruct，截断 tgt，收集 key 集合 ----
    keys = set()
    n_in = 0
    dropped_no_literal = 0
    len_before, len_after, len_literal = [], [], []
    tmp_instruct = []
    with open(args.instruct) as f:
        for line in f:
            d = json.loads(line)
            lit = d.get("literal_tgt", "")
            if not lit:
                dropped_no_literal += 1
                continue
            n_lit = len(word_split(lit))
            tgt_full = d.get("tgt", "")
            tgt_trunc = truncate_to_words(tgt_full, n_lit)

            len_before.append(len(word_split(tgt_full)))
            len_after.append(len(word_split(tgt_trunc)))
            len_literal.append(n_lit)

            d["tgt_full"] = tgt_full
            d["tgt"] = tgt_trunc
            d["trunc_n_words"] = n_lit
            keys.add(d[args.key])
            tmp_instruct.append(d)
            n_in += 1

    with open(args.out_instruct, "w") as fo:
        for d in tmp_instruct:
            fo.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- 2. 读 base，保留 key 在公共集合中的记录 ----
    n_base = 0
    base_keys_seen = Counter()
    with open(args.base) as f, open(args.out_base, "w") as fo:
        for line in f:
            d = json.loads(line)
            k = d.get(args.key)
            if k in keys:
                fo.write(json.dumps(d, ensure_ascii=False) + "\n")
                n_base += 1
                base_keys_seen[k] += 1

    # ---- 3. 报告 ----
    print(f"instruct in        : {args.instruct}")
    print(f"base   in          : {args.base}")
    print(f"out instruct       : {args.out_instruct}")
    print(f"out base (common)  : {args.out_base}")
    print(f"key field          : {args.key}")
    print(f"dropped (no literal): {dropped_no_literal}")
    print(f"instruct written   : {n_in}")
    print(f"base    written    : {n_base}  (matched by {args.key})")
    print()
    print("instruct tgt words  before trunc:", summ(len_before))
    print("instruct tgt words  after  trunc:", summ(len_after))
    print("literal_tgt words              :", summ(len_literal))
    # 校验：截断后应与 literal 等长
    mismatch = sum(1 for a, b in zip(len_after, len_literal) if a != b)
    print(f"length mismatch after truncation: {mismatch} (should be 0)")
    # 校验：base/instruct key 集合是否一致
    missing_in_base = len(keys - set(base_keys_seen.keys()))
    print(f"keys in instruct missing from base: {missing_in_base}")


if __name__ == "__main__":
    main()
