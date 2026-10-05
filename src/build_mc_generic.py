"""Generic multiple-choice anchor traffic from OUTSIDE the pool: 1,500 OpenBookQA training questions and
1,500 HellaSwag validation contexts in the same prompt format and decision position as the pool. Anchoring on these
separates 'same format' from 'same traffic distribution'."""
import json, os
from datasets import load_dataset
from config import DATA as OUT

out = []
ob = load_dataset("allenai/openbookqa", "main", split="train")
for k, r in enumerate(ob.select(range(1500))):
    out.append(dict(id=f"obqa-{k}", source="openbookqa", domain="generic_mc", question=r["question_stem"], choices=r["choices"]["text"], answer="ABCD".index(r["answerKey"])))
hs = load_dataset("Rowan/hellaswag", split="validation")
for k, r in enumerate(hs.select(range(1500))):
    out.append(dict(id=f"hs-{k}", source="hellaswag", domain="generic_mc", question="Choose the most plausible continuation: " + r["ctx"], choices=r["endings"], answer=int(r["label"])))
with open(f"{OUT}/anchors_mc_generic.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(len(out))
