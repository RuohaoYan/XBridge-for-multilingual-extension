#!/usr/bin/env python3
"""Quick test for Plan A: per-token RMS rescaling toward LLM embedding scale."""

import glob
import os
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

TARGET_RMS = 0.009  # empirical mean RMS of LLaMA3 embed_tokens
TARGET_L2 = 0.593   # empirical mean L2 of LLaMA3 embed_tokens

SAMPLES = [
    ("zh", "zho_Hans", "data/flores101/zho_simpl.devtest"),
    ("de", "deu_Latn", "data/flores101/deu.devtest"),
    ("en", "eng_Latn", "data/flores101/eng.devtest"),
]


def load_embed_stats(llm_path):
    for f in sorted(glob.glob(os.path.join(llm_path, "model*.safetensors"))):
        with safe_open(f, framework="pt", device="cpu") as sf:
            key = "model.embed_tokens.weight"
            if key in sf.keys():
                w = sf.get_tensor(key).float()
                rms = (w.pow(2).mean(-1).sqrt())
                return {
                    "global_std": w.std().item(),
                    "rms_mean": rms.mean().item(),
                    "rms_std": rms.std().item(),
                    "l2_mean": w.norm(dim=-1).mean().item(),
                }
    raise FileNotFoundError("embed_tokens not found")


def tensor_stats(h, label):
    rms = (h.float().pow(2).mean(-1).sqrt())
    l2 = h.float().norm(dim=-1)
    return {
        "label": label,
        "mean": h.float().mean().item(),
        "std": h.float().std().item(),
        "rms_mean": rms.mean().item(),
        "rms_std": rms.std().item(),
        "l2_mean": l2.mean().item(),
        "l2_std": l2.std().item(),
    }


def calibrate_rms(h, target_rms=TARGET_RMS):
    rms = h.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
    return h * (target_rms / rms)


def calibrate_l2(h, target_l2=TARGET_L2):
    l2 = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    return h * (target_l2 / l2)


def print_row(s):
    print(
        f"{s['label']:<22} "
        f"std={s['std']:.4f}  "
        f"RMS={s['rms_mean']:.4f}±{s['rms_std']:.4f}  "
        f"L2={s['l2_mean']:.3f}±{s['l2_std']:.3f}"
    )


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mt_path = os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = os.path.join(ROOT, "model/Meta-Llama-3-8B")
    base_model = os.path.join(ROOT, "model/XBridge-base")

    embed_ref = load_embed_stats(llm_path)
    print("=== LLaMA3 embed_tokens reference ===")
    print(embed_ref)
    print(f"Plan A target RMS={TARGET_RMS}, L2={TARGET_L2}\n")

    mt_tok = AutoTokenizer.from_pretrained(mt_path)
    llm_tok = AutoTokenizer.from_pretrained(llm_path)

    config = XBridgeConfig.from_pretrained(base_model)
    config.llm_only = True
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda:0" if device.type == "cuda" else "cpu",
    )
    model.eval()
    encoder = model.model_mt.get_encoder()
    mapping = model.mapping_enc2llm

    print("=== Per-language samples (first sentence each) ===\n")
    for lang, nllb_code, path in SAMPLES:
        line = open(os.path.join(ROOT, path), encoding="utf-8").readline().strip()
        mt_tok.src_lang = nllb_code
        enc = mt_tok(line, return_tensors="pt", add_special_tokens=True)
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)

        with torch.no_grad():
            enc_out = encoder(input_ids=input_ids, attention_mask=attn)[0]
            mapped = mapping(enc_out)

            llm_ids = llm_tok(line, return_tensors="pt", add_special_tokens=False)["input_ids"].to(device)
            llm_emb = model.model_llm.get_input_embeddings()(llm_ids)

            mapped_rms = calibrate_rms(mapped.float())
            mapped_l2 = calibrate_l2(mapped.float())

        print(f"[{lang}] {line[:80]}{'...' if len(line) > 80 else ''}")
        print(f"  tokens: NLLB enc={input_ids.shape[1]}, LLM={llm_ids.shape[1]}")
        for t in [
            tensor_stats(enc_out, "NLLB encoder"),
            tensor_stats(mapped, "mapped (raw)"),
            tensor_stats(mapped_rms, "mapped + RMS cal"),
            tensor_stats(mapped_l2, "mapped + L2 cal"),
            tensor_stats(llm_emb, "LLM embed (same text)"),
        ]:
            print(" ", end="")
            print_row(t)

        raw = tensor_stats(mapped, "raw")
        cal = tensor_stats(mapped_rms, "cal")
        print(
            f"  RMS cal: {raw['rms_mean']:.4f} -> {cal['rms_mean']:.4f} "
            f"(target {TARGET_RMS}, err {abs(cal['rms_mean']-TARGET_RMS):.6f})"
        )
        print()


if __name__ == "__main__":
    main()
