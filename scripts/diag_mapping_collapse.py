"""Diagnose whether baseline step-7000 mapping_enc2llm has collapsed (content-agnostic)
or is merely undertrained.

Feed the SAME diverse zh sentences through NLLB encoder -> mapping, for two mappings:
  (A) baseline checkpoint-7000 (suspected bad)
  (B) XBridge-base (known good, 26.5 BLEU)
Compare between-sentence separability (pairwise cosine of mean-pooled outputs).
Collapse  => outputs nearly identical across sentences (cos -> 1).
Healthy   => outputs clearly separable (cos well below 1), like the raw encoder.
"""
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from safetensors.torch import load_file

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modeling_xbridge import Mapping

DEV = "cuda:0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
N = 16

# --- diverse zh sentences (first N FLORES devtest lines) ---
sents = [l.rstrip("\n") for l in open(f"{ROOT}/data/flores101/zho_simpl.devtest")][:N]

# --- NLLB encoder ---
tok = AutoTokenizer.from_pretrained(f"{ROOT}/model/nllb-200-1.3B")
tok.src_lang = "zho_Hans"
mt = AutoModelForSeq2SeqLM.from_pretrained(f"{ROOT}/model/nllb-200-1.3B", torch_dtype=torch.float32).to(DEV).eval()
enc = mt.get_encoder()

def build_mapping(state):
    m = Mapping(1024, 4096, 4096, 1).to(DEV).float().eval()
    m.load_state_dict({k: v.float() for k, v in state.items()})
    return m

# (A) baseline checkpoint-7000
ck = torch.load(f"{ROOT}/outputs/enc_mt_xen_100k/checkpoint-7000/mapping_enc2llm.pt", map_location="cpu")
map_base_run = build_mapping(ck)

# (B) XBridge-base released mapping (from shard 8)
sh8 = load_file(f"{ROOT}/model/XBridge-base/model-00008-of-00008.safetensors")
xb = {k[len("mapping_enc2llm."):]: v for k, v in sh8.items() if k.startswith("mapping_enc2llm.")}
map_xbridge = build_mapping(xb)

@torch.no_grad()
def pooled(mapping):
    vecs, enc_vecs = [], []
    for s in sents:
        ids = tok(s, return_tensors="pt").to(DEV)
        h = enc(input_ids=ids.input_ids, attention_mask=ids.attention_mask)[0]  # (1,T,1024)
        m = ids.attention_mask.unsqueeze(-1).float()
        enc_pool = (h * m).sum(1) / m.sum(1)            # raw encoder pooled (1,1024)
        y = mapping(h)                                   # (1,T,4096)
        y_pool = (y * m).sum(1) / m.sum(1)               # mapped pooled (1,4096)
        vecs.append(y_pool.squeeze(0))
        enc_vecs.append(enc_pool.squeeze(0))
    return torch.stack(vecs), torch.stack(enc_vecs)

def pairwise_cos(X):
    Xn = F.normalize(X, dim=-1)
    S = Xn @ Xn.T
    iu = torch.triu_indices(len(X), len(X), offset=1)
    v = S[iu[0], iu[1]]
    return v.mean().item(), v.min().item(), v.max().item()

Y_run, ENC = pooled(map_base_run)
Y_xb, _    = pooled(map_xbridge)

print(f"sentences: {N} diverse FLORES zh lines\n")
print("=== between-sentence pairwise cosine of mean-pooled output ===")
print("   (lower = more separable/healthy; ->1.0 = collapsed/content-agnostic)\n")
em, emn, emx = pairwise_cos(ENC)
print(f"  RAW NLLB encoder (reference upper bound): mean={em:.3f}  [{emn:.3f}, {emx:.3f}]")
rm, rmn, rmx = pairwise_cos(Y_run)
print(f"  (A) baseline step-7000 mapping         : mean={rm:.3f}  [{rmn:.3f}, {rmx:.3f}]   norm={Y_run.norm(dim=-1).mean():.2f}")
xm, xmn, xmx = pairwise_cos(Y_xb)
print(f"  (B) XBridge-base mapping (good)         : mean={xm:.3f}  [{xmn:.3f}, {xmx:.3f}]   norm={Y_xb.norm(dim=-1).mean():.2f}")

# reference: Llama token-embedding scale (what the LLM expects as input)
idx = json.load(open(f"{ROOT}/model/XBridge-base/model.safetensors.index.json"))["weight_map"]
emb_shard = idx.get("model.model_llm.model.embed_tokens.weight") or idx.get("model.embed_tokens.weight")
emb_key = "model.model_llm.model.embed_tokens.weight" if "model.model_llm.model.embed_tokens.weight" in idx else None
print("\n=== scale reference ===")
if emb_key:
    W = load_file(f"{ROOT}/model/XBridge-base/{emb_shard}")[emb_key].float()
    print(f"  Llama input-embedding row norm (expected input scale): {W.norm(dim=-1).mean():.2f}")
else:
    cand = [k for k in idx if "embed_tokens" in k]
    print(f"  embed_tokens key candidates: {cand[:4]}")

print("\n=== verdict ===")
gap_run = em - rm  # how much separability the mapping destroys vs encoder
gap_xb  = em - xm
print(f"  separability lost by (A) baseline : {gap_run:+.3f}")
print(f"  separability lost by (B) XBridge   : {gap_xb:+.3f}")
if rm > 0.97 and rm - xm > 0.15:
    print("  => (A) COLLAPSED: baseline mapping output is nearly content-agnostic.")
elif rm - xm < 0.1:
    print("  => NOT collapsed: baseline mapping is about as separable as the good one -> undertrained, not collapsed.")
else:
    print("  => PARTIAL: baseline mapping keeps some content but far less than the good mapping.")
