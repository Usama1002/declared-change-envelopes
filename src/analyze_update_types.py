"""Update types: full fine-tuning, RL updates (decision-token GRPO with in-scope reference KL; GSM8K RLVR with
free generation), and Qwen2.5-72B. Writes results/update_types.json."""
import json, os
import numpy as np
from scipy.stats import binomtest
import analyze as A, updates as U

rng = np.random.RandomState(0)
OUT = {}


def boot_ci(z, B=2000):
    idx = rng.randint(0, len(z), (B, len(z)))
    m = z[idx].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def mcnemar(Da, Db, mask):
    b = int((Da & ~Db)[mask].sum()); c = int((~Da & Db)[mask].sum())
    return float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0


def cert_summary(row):
    """Operative certificate: smallest eps certified on the single audit order fixed in advance (row cond_eps*),
    with the samples it used; the 20-order median at that eps is kept for reference."""
    for e in A.EPS:
        c = row[f"cond_eps{e}"]
        if c["certified"]:
            pc = row[f"perm_cond_eps{e}"]
            return dict(eps=e, n=int(c["n"]), median_n=pc["median_n"], frac=pc["frac_certified"])
    return None


def variant(m, base, tag, u, E, tau):
    v = A.ld(m, tag)
    if v is None:
        return None
    D = A.flips(base, v)
    o = A.CERT & ~E
    z = D[o].astype(float)
    row = A.certificate_row(D, E, dec=base["margin"] > tau, alpha=A.ALPHA / 4)
    r = dict(leakage=float(z.mean()), ci=boot_ci(z), leakage_decisive=float(D[o & (base["margin"] > tau)].mean()),
             in_effect=U.in_envelope_effect(u, A.pool, base, v, mask=A.CERT), cert=cert_summary(row),
             oos_acc_change=float(((v["decision"] == A.ans) & (A.ans >= 0))[o].mean() - ((base["decision"] == A.ans) & (A.ans >= 0))[o].mean()))
    return r, D


GROUPS = {"lora": ["{u}_s0", "{u}_kl1", "{u}_kl10", "{u}_s1", "{u}_kl1s1"],
          "full": ["{u}_fft", "{u}_fftlr3e-6", "{u}_fftkl1", "{u}_fftkl10", "{u}_fftkl1lr3e-6", "{u}_fftkl10lr3e-6", "{u}_fftkl10ab64lr3e-6",
                   "{u}_fftlr1e-6", "{u}_fftkl10ab64lr1e-6", "{u}_fftkl10ab128lr3e-6", "{u}_fftlr3e-6s1", "{u}_fftkl10ab64lr3e-6s1"],
          "grpo": ["{u}_grpo", "{u}_grporef1", "{u}_grpokl1", "{u}_grpos1", "{u}_grporef1s1", "{u}_grpokl1s1"]}
PAIRS = [("{u}_fft", "{u}_fftkl1"), ("{u}_fftlr3e-6", "{u}_fftkl1lr3e-6"), ("{u}_fftlr3e-6", "{u}_fftkl1"), ("{u}_fft", "{u}_fftkl10"),
         ("{u}_grpo", "{u}_grpokl1"), ("{u}_grporef1", "{u}_grpokl1"), ("{u}_grpo", "{u}_grporef1"), ("{u}_s0", "{u}_kl1"), ("{u}_s1", "{u}_kl1s1"), ("{u}_fftlr3e-6", "{u}_fftkl10ab64lr3e-6"),
         ("{u}_fftlr3e-6s1", "{u}_fftkl10ab64lr3e-6s1"), ("{u}_fftlr1e-6", "{u}_fftkl10ab64lr1e-6"), ("{u}_grpos1", "{u}_grpokl1s1"), ("{u}_grporef1s1", "{u}_grpokl1s1")]

for m in ["qwen7b", "llama8b", "qwen72b", "qwen1.5b"]:
    base, null = A.ld(m, "base"), A.ld(m, "null_order1")
    if base is None:
        continue
    tau = A.decisive_tau(base, null) if null is not None else 0.5
    M = OUT.setdefault(m, {})
    if null is not None:
        nf = A.flips(base, null)
        M["floor"] = dict(flip_rate=float(nf[A.CERT].mean()), tau=tau)
    for u in ["med", "policy"]:
        E = A.ENV[u]; Ds = {}
        for g, tags in GROUPS.items():
            for t in tags:
                tag = t.format(u=u); res = variant(m, base, tag, u, E, tau)
                if res is not None:
                    M.setdefault(u, {})[tag] = dict(res[0], group=g); Ds[tag] = res[1]
        for a_, b_ in PAIRS:
            a_, b_ = a_.format(u=u), b_.format(u=u)
            if a_ in Ds and b_ in Ds:
                M[u].setdefault("mcnemar", {})[f"{a_}|{b_}"] = mcnemar(Ds[a_], Ds[b_], A.CERT & ~E)
    # RLVR on GSM8K: declared scope = MMLU mathematics/statistics
    E = U.taxonomy_envelope("math", A.pool); Ds = {}
    for tag in ["math_grpo", "math_grpokl1", "math_grpokl1ff", "math_grpostrong", "math_grpostrongkl1"]:
        res = variant(m, base, tag, "math", E, tau)
        if res is not None:
            meta = json.load(open(f"{A.RES}/{m}/{tag}.json"))
            h = meta.get("history", [])
            res[0]["reward_first10"] = float(np.mean([x["reward"] for x in h[:10]])) if h else None
            res[0]["reward_last10"] = float(np.mean([x["reward"] for x in h[-10:]])) if h else None
            M.setdefault("math", {})[tag] = res[0]; Ds[tag] = res[1]
    for a_, b_ in [("math_grpo", "math_grpokl1"), ("math_grpo", "math_grpokl1ff"), ("math_grpostrong", "math_grpostrongkl1")]:
        if a_ in Ds and b_ in Ds:
            M["math"].setdefault("mcnemar", {})[f"{a_}|{b_}"] = mcnemar(Ds[a_], Ds[b_], A.CERT & ~E)
    for suf in ["_math", "_mathstrong"]:
        ffp = f"{A.RES}/freeform_{m}{suf}.json"
        if os.path.exists(ffp):
            F = json.load(open(ffp))
            M.setdefault("math", {})["freeform" + suf] = F.get("results", F)
        sp = f"{A.RES}/freeform_sem_{m}{suf}.json"
        if os.path.exists(sp):
            M.setdefault("math", {})["freeform_sem" + suf] = json.load(open(sp))

json.dump(OUT, open(f"{A.RES}/update_types.json", "w"), indent=1)
for m, M in OUT.items():
    print(m, "floor", M.get("floor"))
    for u in ["med", "policy", "math"]:
        for tag, r in M.get(u, {}).items():
            if isinstance(r, dict) and "leakage" in r:
                print(f"  {tag:22s} leak {100*r['leakage']:5.1f} [{100*r['ci'][0]:.1f},{100*r['ci'][1]:.1f}] effect {100*r['in_effect']:+5.1f} cert {r['cert']}")
        if "mcnemar" in M.get(u, {}):
            print("   mcnemar", M[u]["mcnemar"])
