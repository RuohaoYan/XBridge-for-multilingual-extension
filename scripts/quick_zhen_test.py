"""Quick zh->en spot-check: load XBridge-base, hot-swap mapping_enc2llm with a
trained checkpoint (in memory, no 38G merge), translate a few sentences."""
import os, sys, torch
from transformers import AutoTokenizer
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modeling_xbridge import XBridgeConfig, LlamaForCasualLMWithXBridge

BASE = f"{ROOT}/model/XBridge-base"
MT = f"{ROOT}/model/nllb-200-1.3B"
LLM = f"{ROOT}/model/Meta-Llama-3-8B"
MAPPING = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/outputs/enc_mt_xen_200k/checkpoint-final/mapping_enc2llm.pt"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 8
# instruction must match training; pass "" for no-instruction models (100k/200k baseline)
INSTRUCTION = sys.argv[3] if len(sys.argv) > 3 else ""

tok_mt = AutoTokenizer.from_pretrained(MT)
tok_llm = AutoTokenizer.from_pretrained(LLM)
tok_llm.pad_token_id = 128002
tok_llm.padding_side = "left"

cfg = XBridgeConfig.from_pretrained(BASE)
cfg.max_gen_len = 128
cfg.llm_only = True
model = LlamaForCasualLMWithXBridge.from_pretrained(
    BASE, config=cfg, torch_dtype=torch.float16, device_map="cuda:0", len_tokenizer_llm=len(tok_llm))
model.model_mt.lm_head.weight = model.model_mt.model.shared.weight
model.eval()
# optionally forbid immediate-EOS to reveal what the model actually generates
_mnt = int(os.environ.get("MIN_NEW_TOKENS", "0"))
if _mnt > 0:
    model.model_llm.generation_config.min_new_tokens = _mnt
    print(f"[min_new_tokens={_mnt}] EOS suppressed for first {_mnt} tokens")

# hot-swap the trained mapping (pass "base" to keep XBridge-base's own mapping = control)
if MAPPING == "base":
    print("[control] using XBridge-base's OWN mapping (no swap)\n")
else:
    state = torch.load(MAPPING, map_location="cpu")
    model.mapping_enc2llm.load_state_dict(state, strict=True)
    print(f"[loaded mapping] {MAPPING}\n")

srcs = [l.rstrip("\n") for l in open(f"{ROOT}/data/flores101/zho_simpl.devtest")][:N]
refs = [l.rstrip("\n") for l in open(f"{ROOT}/data/flores101/eng.devtest")][:N]

inst_ids = tok_llm(INSTRUCTION, add_special_tokens=False)["input_ids"] if INSTRUCTION else []

def translate(text):
    tok_mt.src_lang = "zho_Hans"
    src_ids = tok_mt(text, add_special_tokens=True, return_tensors=None)["input_ids"]  # NLLB source
    # sequence routed by augmentation: 1=source(->encoder), 2=instruction(->LLM embed)
    input_ids = src_ids + inst_ids
    aug = [1] * len(src_ids) + [2] * len(inst_ids)
    input_ids = torch.tensor([input_ids], device="cuda:0")
    attn = torch.ones_like(input_ids)
    aug = torch.tensor([aug], device="cuda:0")
    fds = tok_mt.convert_tokens_to_ids(["zho_Hans"])
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, augmentation=aug,
                    forced_decoder_start_token_id=fds)
    raw = out[0][0].tolist()
    txt = tok_llm.batch_decode(out[0], skip_special_tokens=True)[0].replace("\n", " ")
    return txt, raw

for i, (s, r) in enumerate(zip(srcs, refs)):
    txt, raw = translate(s)
    print(f"[{i+1}] 源  : {s}")
    print(f"    译文: {txt}")
    print(f"    raw : {len(raw)} tok, 前8={raw[:8]}")
    print(f"    参考: {r}\n")
