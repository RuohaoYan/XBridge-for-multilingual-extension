#!/usr/bin/env python3
"""Audit the encoder-only half-continuation corpus.

Checks, for continue_en_half_50k(.jsonl) and its instruct outputs:
  * line / unique-src counts and coverage vs stage-1
  * union(main + shards) coverage -> whether all 50k labels exist
  * main-only src (rows that a shards-only re-merge would silently DROP)
  * format compliance (required fields)
  * the "tgt repeats src" bug (doc section 5) rate
  * empty / degenerate-repetition tgt rate
  * tgt length distribution
  * stage-1 reconstruction (src + ' ' + tgt == orig_tgt)

Usage:
  python scripts/audit_continue_corpus.py
  python scripts/audit_continue_corpus.py --data_dir data/encoder_only --show 5
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))
    return rows


def first_word(s):
    s = s.strip()
    return s.split()[0] if s.split() else ""


def degenerate(tgt, n=4, thresh=3):
    """Flag greedy loops: any 4-word window repeated >= thresh times."""
    words = tgt.split()
    if len(words) < n * thresh:
        return False
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return any(c >= thresh for c in Counter(grams).values())


def audit_instruct(name, rows, show=0):
    src = [r.get("src", "") for r in rows]
    ss = set(src)
    req = {
        "task", "src_lang", "src", "tgt",
        "literal_tgt", "orig_tgt", "label_source", "max_new_tokens",
    }
    miss = sum(1 for r in rows if not req.issubset(r))
    empty = sum(1 for r in rows if not r.get("tgt", "").strip())
    rep = sum(1 for r in rows if r.get("tgt", "").strip().startswith(r.get("src", "").strip()))
    repw = sum(
        1 for r in rows
        if first_word(r.get("tgt", "")) and first_word(r.get("tgt", "")) == first_word(r.get("src", ""))
    )
    startsp = sum(1 for r in rows if r.get("tgt", "").startswith((" ", "\n")))
    degen = sum(1 for r in rows if degenerate(r.get("tgt", "")))
    lens = [len(r.get("tgt", "")) for r in rows]
    n = max(len(rows), 1)

    print(f"\n[{name}]")
    print(f"  lines={len(rows)}  unique_src={len(ss)}  dup_src={len(src)-len(ss)}")
    print(f"  missing_required_fields={miss}  empty_tgt={empty}")
    print(f"  tgt_repeats_full_src(BUG)={rep} ({100*rep/n:.2f}%)"
          f"   tgt_first_word==src_first_word={repw} ({100*repw/n:.2f}%)")
    print(f"  tgt_starts_space/newline={startsp} ({100*startsp/n:.1f}%)"
          f"   degenerate_loop_tgt={degen} ({100*degen/n:.2f}%)")
    if lens:
        l = sorted(lens)
        print(f"  tgt_char_len min={min(lens)} p50={l[len(l)//2]} p95={l[int(len(l)*0.95)]} max={max(lens)}")
    if show:
        print("  suspicious samples (repeats src OR empty OR degenerate):")
        shown = 0
        for r in rows:
            t = r.get("tgt", "")
            if (not t.strip()) or t.strip().startswith(r.get("src", "").strip()) or degenerate(t):
                print(f"    - src={r.get('src','')!r}\n      tgt={t[:120]!r}")
                shown += 1
                if shown >= show:
                    break
        if shown == 0:
            print("    (none)")
    return ss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=os.path.join(ROOT, "data/encoder_only"))
    ap.add_argument("--show", type=int, default=3, help="show N suspicious samples per file")
    args = ap.parse_args()

    D = args.data_dir
    stage1 = os.path.join(D, "continue_en_half_50k.jsonl")
    main_f = os.path.join(D, "continue_en_half_50k_instruct.jsonl")
    shard_glob = os.path.join(D, "continue_en_half_50k_instruct.shard*.jsonl")

    # ---- stage 1 ----
    s1 = load(stage1)
    s1s = set(r["src"] for r in s1)
    recon = sum(1 for r in s1 if (r["src"] + " " + r["tgt"]).strip() == r["orig_tgt"].strip())
    badlen = sum(
        1 for r in s1
        if len(r["src"]) < 3 or len(r["tgt"]) < 3 or len(r["orig_tgt"]) < 8
    )
    print(f"[STAGE1 {os.path.basename(stage1)}]")
    print(f"  lines={len(s1)}  unique_src={len(s1s)}  dup_src={len(s1)-len(s1s)}")
    print(f"  reconstruction(src+' '+tgt==orig_tgt)={recon}/{len(s1)}  min_len_violations={badlen}")

    # ---- instruct outputs ----
    main_src = audit_instruct(os.path.basename(main_f), load(main_f), args.show) if os.path.isfile(main_f) else set()

    shard_sets = []
    all_shard = set()
    for sh in sorted(glob.glob(shard_glob)):
        s = audit_instruct(os.path.basename(sh), load(sh), args.show)
        shard_sets.append(s)
        all_shard |= s

    # ---- cross-file coverage ----
    cnt = Counter()
    for s in shard_sets:
        cnt.update(s)
    overlap = sum(1 for v in cnt.values() if v > 1)
    union = main_src | all_shard

    print("\n[COVERAGE]")
    print(f"  shard_overlap(src in >1 shard, should be 0)={overlap}")
    print(f"  union_shards={len(all_shard)}")
    print(f"  UNION(main+shards)={len(union)}")
    print(f"  of {len(s1s)} stage1 src:  covered={len(s1s & union)}  MISSING={len(s1s - union)}")
    print(f"  main_only (in main, NOT in any shard -> LOST if merge shards-only)={len(main_src - all_shard)}")
    print(f"  outputs_not_in_stage1 (should be 0)={len(union - s1s)}")

    complete = len(s1s - union) == 0
    print("\n[VERDICT]")
    print(f"  all 50k labels generated (union covers stage1): {complete}")
    if len(main_src - all_shard):
        print("  WARNING: re-running merge_continue_instruct_shards.py (shards-only) "
              "would DROP the main-only rows above. Merge must union the existing main file too.")


if __name__ == "__main__":
    main()
