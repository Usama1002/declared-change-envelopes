"""Leakage versus out-of-scope letter-level KL for every trained variant (Remark 1 is exact at the letter level: the
decision is the argmax of the renormalized option-letter distribution). Writes results/kl_leak.json."""
import numpy as np, glob, os, json
import scipy.stats as st
import analyze as A, updates as U

rows = []
for m in ["qwen7b", "llama8b", "qwen32b", "mistral7b", "qwen72b", "qwen1.5b"]:
    base = A.ld(m, "base")
    if base is None:
        continue
    P0 = base["probs"].astype(np.float64)
    for p in sorted(glob.glob(f"{A.RES}/{m}/*.npz")):
        t = os.path.basename(p)[:-4]; u = t.split("_")[0]
        if u not in ("med", "legal", "policy", "edit", "math"):
            continue
        v = A.ld(m, t); E = U.taxonomy_envelope(u, A.pool)
        ev = v["evaluated"] if "evaluated" in v else np.ones(A.N, bool)
        o = A.CERT & ~E & ev
        a = np.clip(P0[o], 1e-6, 1); b = np.clip(v["probs"][o].astype(np.float64), 1e-6, 1)
        K = np.where(P0[o] > 0, a * np.log(a / b), 0).sum(1)
        D = A.flips(base, v)[o]
        srt = np.sort(P0[o], 1); g = srt[:, -1] - srt[:, -2]
        bound = min(((g <= gm).mean() + 2 * K.mean() / gm ** 2) for gm in np.linspace(0.01, 1, 100))
        # training-time diagnostic: the same quantities on the ANCHOR split (available to the developer), used to
        # predict leakage on the audit split
        oa = A.ANC & ~E & ev
        aa_ = np.clip(P0[oa], 1e-6, 1); ba_ = np.clip(v["probs"][oa].astype(np.float64), 1e-6, 1)
        Ka = np.where(P0[oa] > 0, aa_ * np.log(aa_ / ba_), 0).sum(1)
        sa = np.sort(P0[oa], 1); ga = sa[:, -1] - sa[:, -2]
        rows.append(dict(model=m, tag=t, leak=float(D.mean()), K=float(K.mean()), K_median=float(np.median(K)),
                         K_anchor=float(Ka.mean()), bound_anchor=float((Ka >= ga ** 2 / 2).mean()),
                         remark_bound=float(bound), pinsker_pointwise=float((K >= g ** 2 / 2).mean()),
                         leak_given_small_K=float(D[K < 0.01].mean()) if (K < 0.01).any() else None))
json.dump(rows, open(f"{A.RES}/kl_leak.json", "w"), indent=1)
L = np.array([r["leak"] for r in rows]); K = np.array([r["K"] for r in rows])
print(len(rows), "variants; Spearman(leakage, mean out-of-scope letter KL) =", st.spearmanr(L, K))
Ka = np.array([r["K_anchor"] for r in rows]); Ba = np.array([r["bound_anchor"] for r in rows])
print("anchor-split KL vs audit leakage: Spearman", st.spearmanr(L, Ka)[0], " anchor-split bound vs audit leakage:", st.spearmanr(L, Ba)[0])
m6 = L < 0.06
print("below 6%: anchor KL", st.spearmanr(L[m6], Ka[m6])[0], "anchor bound", st.spearmanr(L[m6], Ba[m6])[0])
# how well does the anchor-split bound predict whether the audit leakage is below 5%?
pred = Ba < 0.10
print("anchor-split bound < 10%:", int(pred.sum()), "variants; of these audit leakage < 5%:", int((L[pred] < 0.05).sum()))
print("bound_anchor >= leak on audit for", int((Ba >= L).sum()), "of", len(L))
