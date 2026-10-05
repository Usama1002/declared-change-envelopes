"""Anchor/audit shift and confirmatory audits: (1) anchor/audit domain shift, (2) focal-distillation baseline, (3) confirmatory
certificates for full fine-tuning, GRPO, Qwen2.5-1.5B and Qwen2.5-72B. Writes results/shift_confirm.json."""
import json
import numpy as np
import analyze as A, certify as C, updates as U

FULL_SUB = np.zeros(A.N, bool); FULL_SUB[A.fp.eval_subset()] = True
CERT_ALL = np.zeros(A.N, bool); CERT_ALL[A.split["cert"]] = True
CONF = CERT_ALL & ~FULL_SUB
HO = np.isin(A.dom, ["stem", "humanities_social", "commonsense", "science_qa"])
ALPHA = 0.05 / 4


def cert_single(D, E, mask, alpha=ALPHA, seed=7):
    """One audit order fixed in advance; smallest eps certified by the confidence sequence."""
    ids = np.random.RandomState(seed).permutation(np.where(mask & ~E)[0])
    z = D[ids]
    for eps in [0.005, 0.01, 0.02, 0.05]:
        ok, n, u = C.certify_stream(z, eps, alpha)
        if ok:
            return dict(eps=eps, n=int(n))
    return dict(eps=None, n=len(z), ucb=float(C.ucb_mixture(int(z.sum()), len(z), alpha)))


BOOT = np.random.RandomState(0)


def effect_ci(u, base, v, mask, B=2000):
    """Bootstrap 95% interval of the in-envelope effect (same items and definition as updates.in_envelope_effect)."""
    if u == "policy":
        sel = np.array([r["source"] == "agnews" and r["answer"] == 3 for r in A.pool]) & mask
        x = (v["decision"][sel] == 2).astype(float) - (base["decision"][sel] == 2).astype(float)
    else:
        sel = A.ENV[u] & mask
        x = (v["decision"][sel] == A.ans[sel]).astype(float) - (base["decision"][sel] == A.ans[sel]).astype(float)
    m = x[BOOT.randint(0, len(x), (B, len(x)))].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


OUT = {"shift": {}, "fd": {}, "confirm": {}}
for m in ["qwen7b", "llama8b"]:
    base = A.ld(m, "base")
    for u in ["med", "policy"]:
        E = A.ENV[u]
        # (1) shift: *ho anchors exclude HO domains -> audit HO; *hoB anchors only HO domains -> audit non-HO
        for tag, audit in [("s0", None), ("kl1", None), ("kl10", None), ("kl1ho", HO), ("kl10ho", HO), ("kl1hoB", ~HO), ("kl10hoB", ~HO)]:
            v = A.ld(m, f"{u}_{tag}")
            if v is None:
                continue
            D = A.flips(base, v)
            r = dict(effect=U.in_envelope_effect(u, A.pool, base, v, mask=A.CERT))
            for name, msk in [("HO", HO), ("nonHO", ~HO)]:
                r[f"leak_{name}"] = float(D[A.CERT & ~E & msk].mean())
                r[f"cert_{name}"] = cert_single(D, E, A.CERT & msk)
            if audit is not None:
                r["audit_held_out"] = "HO" if audit is HO else "nonHO"
            OUT["shift"][f"{m}/{u}_{tag}"] = r
        # (2) focal distillation (positive-congruent training) on the training prompts
        for tag in ["s0", "fd1", "fd10", "kl1", "kl10", "grporef1"]:
            v = A.ld(m, f"{u}_{tag}")
            if v is None:
                continue
            D = A.flips(base, v)
            OUT["fd"][f"{m}/{u}_{tag}"] = dict(leak=float(D[A.CERT & ~E].mean()), effect=U.in_envelope_effect(u, A.pool, base, v, mask=A.CERT),
                                              cert=cert_single(D, E, A.CERT))
# (3) confirmatory: models trained anew and evaluated only on the never-used audit half
for m, tags in [("qwen7b", ["cf_med_fftkl10ab64lr1e-6", "cf_policy_fftkl10ab64lr1e-6", "cf_med_grpokl1", "cf_policy_grpokl1"]),
                ("llama8b", ["cf_med_fftkl10ab64lr1e-6", "cf_policy_fftkl10ab64lr1e-6", "cf_med_grpokl1", "cf_policy_grpokl1"]),
                ("qwen1.5b", ["cf_med_kl10", "cf_legal_kl10", "cf_policy_kl10"]),
                ("qwen72b", ["cf_med_kl1", "cf_policy_kl1"])]:
    base = A.ld(m, "base")
    for tag in tags:
        v = A.ld(m, tag)
        if v is None or base is None:
            continue
        u = tag.split("_")[1]; E = A.ENV[u]
        msk = CONF & v["evaluated"]
        D = A.flips(base, v)
        OUT["confirm"][f"{m}/{tag}"] = dict(leak=float(D[msk & ~E].mean()), n=int((msk & ~E).sum()),
                                           effect=U.in_envelope_effect(u, A.pool, base, v, mask=msk), cert=cert_single(D, E, msk),
                                           effect_ci=effect_ci(u, base, v, msk))
json.dump(OUT, open(f"{A.RES}/shift_confirm.json", "w"), indent=1)
for sec, d in OUT.items():
    print("==", sec)
    for k, r in d.items():
        print(" ", k, {a: (round(100 * b, 2) if isinstance(b, float) else b) for a, b in r.items()})
