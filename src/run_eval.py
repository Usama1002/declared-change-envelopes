"""Evaluate a (base or public fine-tuned) model on the traffic pool. Options for null controls."""
import argparse, json, os, time
import fp

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="key in fp.MODELS or HF id")
ap.add_argument("--family", required=True, help="results subdir (base family)")
ap.add_argument("--tag", required=True)
ap.add_argument("--emb", action="store_true")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--order_seed", type=int, default=None)
a = ap.parse_args()
out = f"{fp.RES}/{a.family}"
os.makedirs(out, exist_ok=True)
pool = fp.load_pool()
t0 = time.time()
model, tok = fp.load_model(a.model)
t1 = time.time()
if a.tag.startswith("pub_"):
    res = fp.evaluate_subset(model, tok, pool, fp.eval_subset(), bs=a.bs)
else:
    res = fp.evaluate(model, tok, pool, bs=a.bs, want_emb=a.emb, order_seed=a.order_seed)
fp.save_eval(res, f"{out}/{a.tag}.npz")
json.dump(dict(vars(a), load_time=t1 - t0, eval_time=time.time() - t1), open(f"{out}/{a.tag}.json", "w"))
print("done", a.tag, time.time() - t1, flush=True)
