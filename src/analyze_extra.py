"""Additional statistics (CPU only): bootstrap CIs, cross-update footprint overlap, envelope mass
needed to certify per declaration, direct drift certificates for chains, like-for-like routing vs anchoring.
Writes results/extra.json."""
import json, os, itertools
import numpy as np
import analyze as A
import certify as C
import updates as U

rng = np.random.RandomState(0)
OUT = {}
ans = A.ans


def boot_ci(x, B=1000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return [float("nan")] * 3
    idx = rng.randint(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return [float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def effect_items(u, base, new, mask):
    """Per-item contribution vector whose mean is the in-envelope effect (for paired bootstrap)."""
    ev = new["evaluated"] if "evaluated" in new else np.ones(A.N, bool)
    E = U.taxonomy_envelope(u, A.pool) & ev & mask
    if u in ("med", "legal"):
        sel = E
        return (new["decision"][sel] == ans[sel]).astype(float) - (base["decision"][sel] == ans[sel]).astype(float), sel
    if u == "policy":
        sel = np.array([r["source"] == "agnews" and r["answer"] == 3 for r in A.pool]) & ev & mask
        return (new["decision"][sel] == 2).astype(float) - (base["decision"][sel] == 2).astype(float), sel
    newa = np.array([r.get("new_answer", -1) for r in A.pool])
    return (new["decision"][E] == newa[E]).astype(float) - (base["decision"][E] == newa[E]).astype(float), E


def mcnemar_p(b, c):
    from scipy.stats import binomtest
    n = b + c
    return float(binomtest(min(b, c), n, 0.5).pvalue) if n else 1.0


# 1) bootstrap CIs and paired anchoring effect differences
OUT["ci"] = {}
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    tags = sorted(os.path.basename(p)[:-4] for p in __import__("glob").glob(f"{A.RES}/{model}/*.npz"))
    for tag in tags:
        u = tag.split("_")[0]
        if u not in A.UPD:
            continue
        new = A.ld(model, tag)
        E = A.ENV[u]
        D = A.flips(base, new)
        o = A.CERT & ~E & (new["evaluated"] if "evaluated" in new else True)
        eff, sel = effect_items(u, base, new, A.CERT)
        OUT["ci"][f"{model}/{tag}"] = dict(leakage=boot_ci(D[o]), effect=boot_ci(eff), n_out=int(o.sum()), n_eff=int(sel.sum()))
    # paired differences: anchored vs plain on the same items
    for u in A.UPD:
        b0 = A.ld(model, f"{u}_s0")
        for tag in [f"{u}_kl0.1", f"{u}_kl1", f"{u}_kl10", f"{u}_replay", f"{u}_perm", f"{u}_lr3e-5"]:
            v = A.ld(model, tag)
            if b0 is None or v is None:
                continue
            e0, s0 = effect_items(u, base, b0, A.CERT); e1, s1 = effect_items(u, base, v, A.CERT)
            sel = s0 & s1
            x0, _ = effect_items(u, base, b0, A.CERT & sel)
            x1, _ = effect_items(u, base, v, A.CERT & sel)
            OUT["ci"][f"{model}/{tag}"]["effect_minus_plain"] = boot_ci(x1 - x0)
            # McNemar on out-of-scope leakage (paired over items evaluated by both)
            o = A.CERT & ~A.ENV[u]
            f0 = A.flips(base, b0)[o]; f1 = A.flips(base, v)[o]
            OUT["ci"][f"{model}/{tag}"]["leak_mcnemar_p"] = mcnemar_p(int((f0 & ~f1).sum()), int((~f0 & f1).sum()))

# 2) cross-update footprint overlap (outside both envelopes)
OUT["cross"] = {}
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    runs = {f"{u}_s{s}": A.ld(model, f"{u}_s{s}") for u in ["med", "legal", "policy"] for s in (0, 1)}  # edit excluded: it flips almost everything
    runs = {k: v for k, v in runs.items() if v is not None}
    pubs = {os.path.basename(p)[:-4]: A.ld(model, os.path.basename(p)[:-4]) for p in __import__("glob").glob(f"{A.RES}/{model}/pub_*.npz")}
    allr = {**runs, **pubs}
    fl = {k: A.flips(base, v) for k, v in allr.items()}
    env = {k: (A.ENV[k.split("_")[0]] if k.split("_")[0] in A.UPD else np.zeros(A.N, bool)) for k in allr}
    pairs = []
    for a_, b_ in itertools.combinations(sorted(allr), 2):
        o = A.CERT & ~env[a_] & ~env[b_]
        fa, fb = fl[a_][o], fl[b_][o]
        pa, pb = fa.mean(), fb.mean()
        j = (fa & fb).sum() / max(1, (fa | fb).sum())
        ch = pa * pb / (pa + pb - pa * pb) if (pa + pb) else 0
        same = a_.split("_s")[0] == b_.split("_s")[0] and not a_.startswith("pub")
        pairs.append(dict(a=a_, b=b_, jaccard=float(j), chance=float(ch), same_update=bool(same)))
    OUT["cross"][model] = pairs
    # shared-fragile decomposition: fraction of each seed-0 footprint flipped by >= 2 OTHER (different) updates/public models
    dec = {}
    for u in ["med", "legal", "policy"]:
        k = f"{u}_s0"
        if k not in fl:
            continue
        others = [fl[x] for x in allr if not x.startswith(u + "_")]
        cnt = np.sum(others, 0)
        o = A.CERT & ~A.ENV[u] & fl[k]
        dec[u] = dict(frac_shared_ge2=float((cnt[o] >= 2).mean()), frac_unique=float((cnt[o] == 0).mean()))
    OUT["cross"][model + "_decomposition"] = dec

# 3) envelope mass needed to certify, per declaration score, using the summary scores recomputed quickly
S = json.load(open(f"{A.RES}/summary.json"))
OUT["mass_to_certify"] = {}
for model in ["qwen7b", "llama8b"]:
    for u in A.UPD:
        e = S[model]["updates"].get(u)
        if not e:
            continue
        rec = {}
        for k, d in e["declarations"].items():
            ms = [p["mass"] for p in d["curve"] if p["leakage"] <= 0.02]
            ms5 = [p["mass"] for p in d["curve"] if p["leakage"] <= 0.05]
            rec[k] = dict(mass_leak_le_2=min(ms) if ms else None, mass_leak_le_5=min(ms5) if ms5 else None)
        OUT["mass_to_certify"][f"{model}/{u}"] = rec

# 4) direct end-to-end drift certificates for chains (single CS on drift outside the union at each T)
OUT["chains_direct"] = {}
for cm in ["qwen7b", "llama8b"]:
    base = A.ld(cm, "base")
    for ch, tags in [("plain", ["med_s0", "chain_plain_2", "chain_plain_3", "chain_plain_4"]),
                     ("kl", ["chain_kl_1", "chain_kl_2", "chain_kl_3", "chain_kl_4"]),
                     ("plain_mix", ["med_s0", "chain_plain_2", "chain_plain_3", "chain_plain_4mix"]),
                     ("kl_mix", ["chain_kl_1", "chain_kl_2", "chain_kl_3", "chain_kl_4mix"])]:
        seq = [A.ld(cm, t) for t in tags]
        if any(s is None for s in seq):
            continue
        union = np.zeros(A.N, bool); rows = []
        for k, u in enumerate(A.UPD):
            union |= A.ENV[u]
            z = (A.flips(base, seq[k]) & ~union)[A.CERT]
            rows.append(dict(T=k + 1, observed=float(z.mean()), direct_ucb=C.ucb_mixture(int(z.sum()), len(z), C.spend(0.05, k + 1))))
        OUT["chains_direct"][ch if cm == "qwen7b" else f"{cm}_{ch}"] = rows

# 5) like-for-like routing: envelope = NL judge (mass matched to taxonomy); compare router vs anchored model on the same envelope
judge = np.load(f"{A.RES}/judge_qwen7b.npz")
OUT["routing"] = {}
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    for u in A.UPD:
        thr = np.quantile(judge[u][A.ANC], 1 - A.ENV[u][A.ANC].mean())
        J = judge[u] >= thr
        row = {}
        for tag in [f"{u}_s0", f"{u}_kl1", f"{u}_kl10"]:
            v = A.ld(model, tag)
            if v is None:
                continue
            D = A.flips(base, v)
            gated = dict(decision=np.where(J, v["decision"], base["decision"]), evaluated=v.get("evaluated", np.ones(A.N, bool)))
            row[tag] = dict(leak_vs_judge_env=float(D[A.CERT & ~J].mean()), effect=U.in_envelope_effect(u, A.pool, base, v, mask=A.CERT),
                            router_effect=U.in_envelope_effect(u, A.pool, base, gated, mask=A.CERT))
        OUT["routing"][f"{model}/{u}"] = row

# 6) anchoring on the judge envelope itself (declared E = NL-judge envelope, anchors = anchor items outside it)
OUT["judge_anchor"] = {}
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    for u in A.UPD:
        v = A.ld(model, f"judgeenv_{u}_kl1")
        if v is None:
            continue
        thr = np.quantile(judge[u][A.ANC], 1 - A.ENV[u][A.ANC].mean())
        J = judge[u] >= thr
        D = A.flips(base, v)
        row = dict(leak_vs_judge_env=float(D[A.CERT & ~J].mean()), leak_vs_taxonomy=float(D[A.CERT & ~A.ENV[u]].mean()),
                   effect=U.in_envelope_effect(u, A.pool, base, v, mask=A.CERT),
                   cert_judge=A.certificate_row(D, J, alpha=0.05 / 4, n_perm=20))
        row["cert_judge"] = {k: row["cert_judge"][k] for k in ["perm_cond_eps0.02", "perm_cond_eps0.05", "ucb_full_cond"]}
        OUT["judge_anchor"][f"{model}/{u}"] = row

# 7) edit update: LoRA variants vs locality-preserving editors, with the neighborhood slice reported separately
OUT["edit_compare"] = {}
NB = A.dom == "entity_facts_neighbor"
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base"); E = A.ENV["edit"]
    for tag in ["edit_s0", "edit_kl1", "edit_mix", "edit_mixkl1", "edit_memit", "edit_memitw60k", "edit_alphaedit", "edit_memitw2k"]:
        v = A.ld(model, tag)
        if v is None:
            continue
        D = A.flips(base, v); o = A.CERT & ~E
        acc = lambda d, mm: float((d["decision"][mm] == ans[mm]).mean())
        ev = o & (ans >= 0)
        z = D[np.random.RandomState(7).permutation(np.where(o)[0])]
        ok, n = C.certify_onesided(z, 0.05, 0.05 / 16)
        OUT["edit_compare"][f"{model}/{tag}"] = dict(leak=float(D[o].mean()), leak_ci=boot_ci(D[o]), leak_excl_neighbors=float(D[o & ~NB].mean()),
                                                    neighbor_flip=float(D[A.CERT & NB].mean()), n_neighbor=int((A.CERT & NB).sum()),
                                                    effect=U.in_envelope_effect("edit", A.pool, base, v, mask=A.CERT),
                                                    oos_acc_change=acc(v, ev) - acc(base, ev), oos_acc=acc(v, ev),
                                                    cert5_onesided=bool(ok), cert5_n=int(n),
                                                    ucb_mix=C.ucb_mixture(int(D[o].sum()), int(o.sum()), 0.05 / 4))

# 8) second update order for the Qwen chains (policy, edit with true facts, legal, med), plain and anchored
OUT["chains_order2"] = {}
ORD2 = ["policy", "edit", "legal", "med"]
base = A.ld("qwen7b", "base")
for ch, tags in [("plain", ["policy_s0", "c2_plain_2", "c2_plain_3", "c2_plain_4"]), ("kl", ["c2_kl_1", "c2_kl_2", "c2_kl_3", "c2_kl_4"])]:
    seq = [A.ld("qwen7b", t) for t in tags]
    if any(x is None for x in seq):
        continue
    union = np.zeros(A.N, bool); rows = []; prev = base; cb = 0.0
    for k, u in enumerate(ORD2):
        union |= A.ENV[u]
        z = (A.flips(base, seq[k]) & ~union)[A.CERT]
        st = (A.flips(prev, seq[k]) & ~A.ENV[u])[A.CERT]
        cb += C.ucb_mixture(int(st.sum()), len(st), C.spend(0.05, k + 1))
        rows.append(dict(T=k + 1, update=u, observed=float(z.mean()), certified_bound=cb,
                         direct_ucb=C.ucb_mixture(int(z.sum()), len(z), C.spend(0.05, k + 1)),
                         in_effect=U.in_envelope_effect(u, A.pool, prev, seq[k], mask=A.CERT)))
        prev = seq[k]
    OUT["chains_order2"][ch] = rows

# 9) equivalence (TOST) for effect preservation: anchored minus plain, margin = half of the plain effect
OUT["tost"] = {}
for model in ["qwen7b", "llama8b"]:
    base = A.ld(model, "base")
    for u in ["med", "legal", "policy"]:
        b0 = A.ld(model, f"{u}_s0")
        for tag in [f"{u}_kl1", f"{u}_kl10"]:
            v = A.ld(model, tag)
            if v is None or b0 is None:
                continue
            x0, s0 = effect_items(u, base, b0, A.CERT); x1, s1 = effect_items(u, base, v, A.CERT)
            if not np.array_equal(s0, s1):
                continue
            d = x1 - x0
            margin = 0.5 * abs(x0.mean())
            idx = rng.randint(0, len(d), (2000, len(d))); bm = d[idx].mean(1)
            lo, hi = np.percentile(bm, 5), np.percentile(bm, 95)   # 90% interval = TOST at 5%
            OUT["tost"][f"{model}/{tag}"] = dict(diff=float(d.mean()), ci90=[float(lo), float(hi)], margin=float(margin),
                                                 equivalent=bool(lo > -margin and hi < margin), noninferior=bool(lo > -margin))

json.dump(OUT, open(f"{A.RES}/extra.json", "w"), indent=1)
print("wrote extra.json")
