"""Base-model embeddings of each update's training set (for data-derived kNN envelopes)."""
import argparse, numpy as np
import fp, updates as U

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
a = ap.parse_args()
model, tok = fp.load_model(a.model)
out = {}
for u in ["med", "legal", "policy", "edit"]:
    R, _ = U.train_set(u)
    out[u] = fp.evaluate(model, tok, R, bs=64, want_emb=True)["emb"]
    print(u, out[u].shape, flush=True)
np.savez_compressed(f"{fp.RES}/{a.model}/train_emb.npz", **out)
