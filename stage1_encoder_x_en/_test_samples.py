#!/usr/bin/env python3
"""One-off: load the trained mapping_enc2llm once and translate a few sentences
across several languages to eyeball quality. Not part of the training pipeline."""
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling_xbridge import LlamaForCasualLMWithXBridge, XBridgeConfig  # noqa: E402
from stage1_encoder_x_en.infer_encoder_x_en import build_batch  # reuse batch builder

BASE_MODEL = "outputs/stage1_encoder_x_en_multi/checkpoint-final"
MT_PATH = "model/nllb-200-1.3B"
LLM_PATH = "model/Meta-Llama-3-8B"
PROMPT = "Translate into English:"
MAX_NEW_TOKENS = 64

# (nllb code, source sentence, reference gloss for our own judgement)
SAMPLES = [
    ("zho_Hans", "今天天气很好，我想去公园散步。", "weather nice, want to walk in the park"),
    ("deu_Latn", "Ich habe gestern ein interessantes Buch gelesen.", "I read an interesting book yesterday"),
    ("fra_Latn", "Le train part à huit heures demain matin.", "the train leaves at eight tomorrow morning"),
    ("rus_Cyrl", "Она любит слушать музыку по вечерам.", "she likes listening to music in the evenings"),
    ("jpn_Jpan", "彼は毎朝コーヒーを飲みます。", "he drinks coffee every morning"),
    ("swh_Latn", "Watoto wanacheza mpira uwanjani.", "children playing ball on the field"),
]

tok_mt = AutoTokenizer.from_pretrained(MT_PATH)
tok_llm = AutoTokenizer.from_pretrained(LLM_PATH)
tok_llm.pad_token_id = 128002
tok_llm.padding_side = "left"

config = XBridgeConfig.from_pretrained(BASE_MODEL)
config.mt_path, config.llm_path = MT_PATH, LLM_PATH
config.llm_only = True
config.max_gen_len = MAX_NEW_TOKENS
config.freeze_enc = config.freeze_llm = config.freeze_dec = True
config.freeze_mapping_enc2llm = False
config.freeze_mapping_llm2dec = True
config.dec_lambda = 0.0
config.ot_lambda = 0.0

model = LlamaForCasualLMWithXBridge(config, is_training=True, len_tokenizer_llm=len(tok_llm))
model.mapping_enc2llm.load_state_dict(
    torch.load(Path(BASE_MODEL) / "mapping_enc2llm.pt", map_location="cpu"), strict=True)
model.eval()
device = next(model.mapping_enc2llm.parameters()).device
print(f"loaded {BASE_MODEL} on {device}\n" + "=" * 70, flush=True)

with torch.no_grad():
    for nllb, sentence, gloss in SAMPLES:
        batch = build_batch([sentence], nllb, PROMPT, tok_mt, tok_llm)
        batch = {k: v.to(device) for k, v in batch.items()}
        out = tok_llm.batch_decode(model(**batch)[0], skip_special_tokens=True)[0]
        print(f"[{nllb}] SRC : {sentence}")
        print(f"          OUT : {out.replace(chr(10), ' ').strip()}")
        print(f"          (ref: {gloss})")
        print("-" * 70, flush=True)
