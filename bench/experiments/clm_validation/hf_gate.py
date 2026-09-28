"""CLM acceptance gate without vLLM: Qwen3-8B last-token embeddings from plain
transformers, fed to the released heads through clm's own Engine. If this also
misses the published reference numbers, the mismatch is between the docs and the
released head, not in our serving stack.

    .venv-clm/bin/python bench/experiments/clm_validation/hf_gate.py   (on the RTX 3090 box, with vLLM stopped)
"""
import json, numpy as np, torch
from transformers import AutoModel, AutoTokenizer
from clm.engine import Engine
from clm.embedder import l2


class HFEmbedder:
    def __init__(self, name="Qwen/Qwen3-8B"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.m = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def embed(self, texts):
        out = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
            h = self.m(input_ids=ids).last_hidden_state[0, -1].float().cpu().numpy()
            out.append(l2(h))
        return np.stack(out), 0


eng = Engine(embedder=HFEmbedder())
state = "Customer: my invoice was charged twice and nobody answers the phone!"
qs = {"urgency": {"type": "noul", "instructions": "Is this urgent?"},
      "department": {"type": "choice", "instructions": "Which team should handle this?",
                     "criteria": {"billing": "Charges, invoices, refunds", "technical": "Bugs and outages"}},
      "frustration": {"type": "score", "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Frustrated", "Very angry"]}}
r = eng.answer(state, qs)
r = r.get("answers", r)
got = {"urgency": r["urgency"]["noul"], "billing": r["department"]["probabilities"]["billing"],
       "score": r["frustration"]["score"]}
exp = {"urgency": 0.41022, "billing": 0.93878, "score": 1.98386}
for k in exp:
    print(f"{k:8s} expected {exp[k]:.5f}  transformers {got[k]:.5f}  |d| {abs(exp[k]-got[k]):.1e}")
