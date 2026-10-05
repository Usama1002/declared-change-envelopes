"""Confirmatory audit on the never-evaluated half of the audit split, seed replicates of anchored updates, and the
plain-LoRA learning-rate x epoch sweep. Writes results/confirm.json."""
import json, glob, os
import numpy as np
import analyze as A
import certify as C
import updates as U

FULL_SUB = np.zeros(A.N, bool); FULL_SUB[A.fp.eval_subset()] = True
CERT_ALL = np.zeros(A.N, bool); CERT_ALL[A.split["cert"]] = True
CONF = CERT_ALL & ~FULL_SUB          # never evaluated by any update during development
DEV = CERT_ALL & FULL_SUB            # the development audit set used in the main tables
OUT = {"n_conf": int(CONF.sum()), "n_dev": int(DEV.sum())}


def cert_single(D, E, mask, alpha, seed=7):
    """One pre-registered random order; smallest eps certified (conditional form)."""
    ids = np.where(mask & ~E)[0]
    ids = np.random.RandomState(seed).permutation(ids)
    z = D[ids]
    for eps in [0.005, 0.01, 0.02, 0.05]:
        ok, n, u = C.certify_stream(z, eps, alpha)
        if ok:
            return dict(eps=eps, n=n)
    return dict(eps=None, n=len(z), ucb=C.ucb_mixture(int(z.sum()), len(z), alpha))


for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    OUT[model] = {"plain_conf": {}, "anchored": {}, "sweep": {}}
    for u in A.UPD:
        E = A.ENV[u]
        for s in (0, 1):
            r = A.ld(model, f"comp_{u}_s{s}")
            if r is None:
                continue
            D = A.flips(base, r)
            OUT[model]["plain_conf"][f"{u}_s{s}"] = dict(leak_conf=float(D[CONF & ~E].mean()), n=int((CONF & ~E).sum()),
                                                         effect_conf=U.in_envelope_effect(u, A.pool, base, r, mask=CONF))
    for p in sorted(glob.glob(f"{A.RES}/{model}/rep_*.npz")):
        tag = os.path.basename(p)[:-4]          # rep_{u}_kl{b}_seed{s}
        _, u, kl, seed = tag.split("_")
        r = A.ld(model, tag); E = A.ENV[u]
        D = A.flips(base, r)
        key = f"{u}_{kl}"
        ent = OUT[model]["anchored"].setdefault(key, {"seeds": {}})
        ent["seeds"][seed] = dict(leak_dev=float(D[DEV & ~E].mean()), leak_conf=float(D[CONF & ~E].mean()),
                                  leak_dev_decisive=float(D[DEV & ~E & (base["margin"] > 0.4)].mean()),
                                  effect_dev=U.in_envelope_effect(u, A.pool, base, r, mask=DEV),
                                  effect_conf=U.in_envelope_effect(u, A.pool, base, r, mask=CONF),
                                  cert_conf=cert_single(D, E, CONF, 0.05 / 4))
    # add the development seed-0 anchored variant for the seed summary
    for key, ent in OUT[model]["anchored"].items():
        v0 = A.ld(model, key)
        if v0 is not None:
            u = key.split("_")[0]; E = A.ENV[u]; D = A.flips(base, v0)
            ent["seeds"]["seed0"] = dict(leak_dev=float(D[DEV & ~E].mean()), effect_dev=U.in_envelope_effect(u, A.pool, base, v0, mask=DEV))
        L = [x["leak_dev"] for x in ent["seeds"].values()]; Ef = [x["effect_dev"] for x in ent["seeds"].values()]
        ent["summary"] = dict(n_seeds=len(L), leak_mean=float(np.mean(L)), leak_sd=float(np.std(L, ddof=1)) if len(L) > 1 else None,
                              effect_mean=float(np.mean(Ef)), effect_sd=float(np.std(Ef, ddof=1)) if len(Ef) > 1 else None)
    for p in sorted(glob.glob(f"{A.RES}/{model}/sweep_*.npz")):
        tag = os.path.basename(p)[:-4]
        u = tag.split("_")[1]; E = A.ENV[u]
        r = A.ld(model, tag); D = A.flips(base, r)
        OUT[model]["sweep"][tag] = dict(leak=float(D[DEV & ~E].mean()), effect=U.in_envelope_effect(u, A.pool, base, r, mask=DEV))
json.dump(OUT, open(f"{A.RES}/confirm.json", "w"), indent=1)
print("wrote confirm.json", OUT["n_conf"], OUT["n_dev"])
