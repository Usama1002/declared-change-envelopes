"""Effect preservation with adequate power: evaluate plain and anchored medical adapters on ALL medical
queries of the pool (6,277; the update evaluations used only half) plus MedQA-USMLE test (1,273; never used anywhere),
and report accuracy gains over the incumbent with paired bootstrap intervals for anchored minus plain."""
import argparse, json, numpy as np
import fp
from datasets import load_dataset
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapters", required=True, help="comma list name=checkpoint_suffix")
a = ap.parse_args()
pool = fp.load_pool()
med = [r for r in pool if r["domain"] == "medical"]
mq = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
medqa = [dict(question=r["question"], choices=[r["options"][k] for k in "ABCD"], answer="ABCD".index(r["answer_idx"])) for r in mq]
SETS = {"pool_medical": med, "medqa": medqa}
gold = {k: np.array([r["answer"] for r in v]) for k, v in SETS.items()}
model, tok = fp.load_model(a.model)
corr = {}
for k, v in SETS.items():
    corr[f"base/{k}"] = fp.evaluate(model, tok, v, bs=32)["decision"] == gold[k]
names = []
for item in a.adapters.split(","):
    name, suf = item.split("=")
    m = PeftModel.from_pretrained(model, f"{fp.CKPT}/{a.model}_{suf}")
    m.eval()
    for k, v in SETS.items():
        corr[f"{name}/{k}"] = fp.evaluate(m, tok, v, bs=32)["decision"] == gold[k]
    model = m.unload(); names.append(name)
    print(name, {k: round(float(corr[f"{name}/{k}"].mean() - corr[f"base/{k}"].mean()), 4) for k in SETS}, flush=True)

rng = np.random.RandomState(0)


def boot(x, B=2000):
    idx = rng.randint(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


out = {"n": {k: len(v) for k, v in SETS.items()}, "base_acc": {k: float(corr[f"base/{k}"].mean()) for k in SETS}, "gain": {}, "vs_plain": {}}
for n in names:
    for k in SETS:
        out["gain"][f"{n}/{k}"] = boot(corr[f"{n}/{k}"].astype(float) - corr[f"base/{k}"].astype(float))
plains = [n for n in names if n.startswith("plain")]
for n in names:
    if n.startswith("plain"):
        continue
    for k in SETS:
        # anchored minus the mean of the plain seeds, paired over items
        pm = np.mean([corr[f"{p}/{k}"].astype(float) for p in plains], 0)
        out["vs_plain"][f"{n}/{k}"] = boot(corr[f"{n}/{k}"].astype(float) - pm)
json.dump(out, open(f"{fp.RES}/effect_{a.model}.json", "w"), indent=1)
print(json.dumps(out["vs_plain"], indent=0))
