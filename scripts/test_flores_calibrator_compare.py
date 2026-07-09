#!/usr/bin/env python3
"""Compare FLORES x->en translation with/without Plan-A embedding-scale calibration."""

import os
import sys

import sacrebleu
import torch
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

TARGET_RMS = 0.009
TARGET_L2 = 0.593
MODES = ("none", "rms", "l2")
LANGS = ("zh", "de", "ja")
SAMPLE_IDS = (0, 100, 200, 300, 400)

LANG_MAP = {
    "zh": ("Chinese", "zho_Hans", "zho_simpl"),
    "de": ("German", "deu_Latn", "deu"),
    "ja": ("Japanese", "jpn_Jpan", "jpn"),
}


def calibrate(h, mode):
    if mode == "none":
        return h
    if mode == "rms":
        rms = h.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-6)
        return h * (TARGET_RMS / rms)
    if mode == "l2":
        l2 = h.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        return h * (TARGET_L2 / l2)
    raise ValueError(mode)


def install_calibrator(model, mode):
    mapping = model.mapping_enc2llm
    original = mapping.forward

    def forward(hidden_states):
        return calibrate(original(hidden_states), mode)

    mapping.forward = forward
    return original


def restore_calibrator(model, original):
    model.mapping_enc2llm.forward = original


def translate_batch(model, tokenizer_mt, tokenizer_llm, lines, src_name, nllb_code, device):
    tokenizer_mt.src_lang = nllb_code
    input_ids_mt = [
        tokenizer_mt(text, add_special_tokens=True, return_tensors=None)["input_ids"]
        for text in lines
    ]
    pad_id = tokenizer_llm.pad_token_id
    max_len = max(len(x) for x in input_ids_mt)
    batch = []
    mask = []
    aug = []
    for seq in input_ids_mt:
        pad = [pad_id] * (max_len - len(seq)) + seq
        batch.append(pad)
        mask.append([0] * (max_len - len(seq)) + [1] * len(seq))
        aug.append([0] * (max_len - len(seq)) + [1] * len(seq))
    input_ids = torch.tensor(batch, device=device)
    attention_mask = torch.tensor(mask, device=device)
    augmentation = torch.tensor(aug, device=device)
    forced = tokenizer_mt.convert_tokens_to_ids(["eng_Latn"])

    with torch.no_grad():
        llm_ids, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            augmentation=augmentation,
            forced_decoder_start_token_id=forced,
        )
    return [
        tokenizer_llm.decode(ids, skip_special_tokens=True).replace("\n", " ").strip()
        for ids in llm_ids
    ]


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mt_path = os.path.join(ROOT, "model/nllb-200-1.3B")
    llm_path = os.path.join(ROOT, "model/Meta-Llama-3-8B")
    base_model = os.path.join(ROOT, "model/XBridge-base")
    ref_path = os.path.join(ROOT, "data/flores101/eng.devtest")
    refs_all = [l.rstrip("\n") for l in open(ref_path, encoding="utf-8")]

    tokenizer_mt = AutoTokenizer.from_pretrained(mt_path)
    tokenizer_llm = AutoTokenizer.from_pretrained(llm_path)
    tokenizer_llm.pad_token_id = 128002
    tokenizer_llm.padding_side = "left"

    config = XBridgeConfig.from_pretrained(base_model)
    config.max_gen_len = 256
    config.llm_only = True
    print("Loading XBridge-base...")
    model = LlamaForCasualLMWithXBridge.from_pretrained(
        base_model,
        config=config,
        torch_dtype=torch.float16,
        device_map="cuda:0" if device.type == "cuda" else "cpu",
        len_tokenizer_llm=len(tokenizer_llm),
    )
    model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
    model.eval()

    original_forward = model.mapping_enc2llm.forward
    results = {}

    for lang in LANGS:
        src_name, nllb_code, flores_code = LANG_MAP[lang]
        src_path = os.path.join(ROOT, f"data/flores101/{flores_code}.devtest")
        sources = [l.rstrip("\n") for l in open(src_path, encoding="utf-8")]
        ids = [i for i in SAMPLE_IDS if i < len(sources)]
        src_lines = [sources[i] for i in ids]
        ref_lines = [refs_all[i] for i in ids]

        print(f"\n=== {lang}->en ({len(ids)} samples) ===")
        for mode in MODES:
            install_calibrator(model, mode)
            hyps = translate_batch(
                model, tokenizer_mt, tokenizer_llm, src_lines, src_name, nllb_code, device
            )
            bleu = sacrebleu.corpus_bleu(hyps, [ref_lines], tokenize="flores200").score
            results[(lang, mode)] = (bleu, hyps, ref_lines, src_lines)
            print(f"  [{mode:4s}] BLEU = {bleu:.2f}")

        restore_calibrator(model, original_forward)

        # show one example where calibration changes output most
        _, hyps_none, _, _ = results[(lang, "none")]
        _, hyps_rms, refs, srcs = results[(lang, "rms")]
        best_i = 0
        for i, (a, b) in enumerate(zip(hyps_none, hyps_rms)):
            if a != b:
                best_i = i
                break
        print(f"\n  Example #{ids[best_i]} ({lang}->en):")
        print(f"  SRC: {srcs[best_i][:120]}...")
        print(f"  REF: {refs[best_i][:120]}...")
        print(f"  none: {hyps_none[best_i][:120]}...")
        print(f"  rms : {hyps_rms[best_i][:120]}...")
        print(f"  l2  : {results[(lang, 'l2')][1][best_i][:120]}...")

    print("\n=== Summary (x->en BLEU, flores200) ===")
    print(f"{'lang':<6} " + " ".join(f"{m:>8}" for m in MODES))
    for lang in LANGS:
        row = f"{lang:<6} " + " ".join(f"{results[(lang, m)][0]:8.2f}" for m in MODES)
        print(row)

    avg = {m: sum(results[(l, m)][0] for l in LANGS) / len(LANGS) for m in MODES}
    print(f"{'avg':<6} " + " ".join(f"{avg[m]:8.2f}" for m in MODES))


if __name__ == "__main__":
    main()
