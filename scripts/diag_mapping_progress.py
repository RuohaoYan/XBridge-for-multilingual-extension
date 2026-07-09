"""Does more data move the trained mapping TOWARD XBridge-base's (good) solution?

Compare three mapping_enc2llm on the same zh sentences (NLLB encoder -> mapping):
  base = XBridge-base (good, 26.5 BLEU) | 100k = enc_mt_xen_100k | 200k = enc_mt_xen_200k
Metrics:
  (1) between-sentence spread  = mean pairwise cosine of mean-pooled outputs
      (base sits at ~0.907; 100k was ~0.758)
  (2) closeness-to-base        = per-sentence cosine(trained_output, base_output)
      higher => the trained mapping's actual output is nearer base's known-good output.
If 200k > 100k on (2), scaling data is the lever. If ~equal, it is not.
"""
import os, sys, torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from safetensors.torch import load_file

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from modeling_xbridge import Mapping

DEV = "cuda:0"
N = 24
sents = [l.rstrip("\n") for l in open(f"{ROOT}/data/flores101/zho_simpl.devtest")][:N]

tok = AutoTokenizer.from_pretrained(f"{ROOT}/model/nllb-200-1.3B"); tok.src_lang = "zho_Hans"
mt = AutoModelForSeq2SeqLM.from_pretrained(f"{ROOT}/model/nllb-200-1.3B", torch_dtype=torch.float32).to(DEV).eval()
enc = mt.get_encoder()

def build(state):
    m = Mapping(1024, 4096, 4096, 1).to(DEV).float().eval()
    m.load_state_dict({k: v.float() for k, v in state.items()})
    return m

sh8 = load_file(f"{ROOT}/model/XBridge-base/model-00008-of-00008.safetensors")
maps = {
    "base": build({k[len('mapping_enc2llm.'):]: v for k, v in sh8.items() if k.startswith('mapping_enc2llm.')}),
    "100k": build(torch.load(f"{ROOT}/outputs/enc_mt_xen_100k/checkpoint-7000/mapping_enc2llm.pt", map_location="cpu")),
    "200k": build(torch.load(f"{ROOT}/outputs/enc_mt_xen_200k/checkpoint-final/mapping_enc2llm.pt", map_location="cpu")),
}

@torch.no_grad()
def pooled(mapping):
    out = []
    for s in sents:
        ids = tok(s, return_tensors="pt").to(DEV)
        h = enc(input_ids=ids.input_ids, attention_mask=ids.attention_mask)[0]
        m = ids.attention_mask.unsqueeze(-1).float()
        out.append((mapping(h) * m).sum(1).squeeze(0) / m.sum())
    return torch.stack(out)

pool = {k: pooled(v) for k, v in maps.items()}

def spread(X):
    Xn = F.normalize(X, dim=-1); S = Xn @ Xn.T
    iu = torch.triu_indices(len(X), len(X), 1)
    return S[iu[0], iu[1]].mean().item()

def closeness(X, B):
    return F.cosine_similarity(X, B, dim=-1).mean().item()

print(f"sentences: {N}\n")
print(f"{'mapping':6} {'spread(cos)':>12} {'norm':>8} {'closeness-to-base':>18}")
for k in ["base", "100k", "200k"]:
    print(f"{k:6} {spread(pool[k]):>12.3f} {pool[k].norm(dim=-1).mean():>8.2f} "
          f"{closeness(pool[k], pool['base']):>18.3f}")

c100 = closeness(pool['100k'], pool['base'])
c200 = closeness(pool['200k'], pool['base'])
print("\n=== verdict ===")
print(f"closeness-to-base: 100k={c100:.3f} -> 200k={c200:.3f}  (delta {c200-c100:+.3f})")
if c200 - c100 > 0.03:
    print("=> 200k moved TOWARD base: scaling data IS the lever (need much more, multilingual).")
elif abs(c200 - c100) <= 0.03:
    print("=> 200k barely moved despite 2x data: scale is NOT the lever -> recipe/objective issue.")
else:
    print("=> 200k moved AWAY from base: more of this data is counterproductive.")
