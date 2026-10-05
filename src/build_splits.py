"""Merge CounterFact items into the pool (once), then build the anchor/audit split and the fixed evaluation half.
Seeds match the ones used for all reported results."""
import json, os, random
from config import DATA

os.chdir(DATA)
P = [json.loads(l) for l in open("pool.jsonl")]
if not any(r["source"].startswith("cf_") for r in P):
    P += [json.loads(l) for l in open("pool_cf.jsonl")]
    random.Random(1).shuffle(P)
    with open("pool.jsonl", "w") as f:
        for r in P:
            f.write(json.dumps(r) + "\n")
n = len(P)
idx = list(range(n)); random.Random(2).shuffle(idx)
json.dump({"anchor": sorted(idx[:int(.3 * n)]), "cert": sorted(idx[int(.3 * n):])}, open("split.json", "w"))
idx = list(range(n)); random.Random(3).shuffle(idx)
json.dump(sorted(idx[:n // 2]), open("subset.json", "w"))
print(n)
