"""Compute all footprint metrics, declarations, certificates, composition, and public fine-tune footprints.
Writes results/summary.json."""
import json, os, glob, math, random
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import fp, updates as U, certify as C

RES = fp.RES
pool = fp.load_pool()
N = len(pool)
split = json.load(open(f"{fp.DATA}/split.json"))
SUB = np.zeros(N, bool); SUB[fp.eval_subset()] = True  # all update evaluations use this fixed half of the pool
ANC = np.zeros(N, bool); ANC[split["anchor"]] = True
CERT = ~ANC & SUB
ANC = ANC & SUB
ans = np.array([r["answer"] for r in pool])
dom = np.array([r["domain"] for r in pool])
UPD = ["med", "legal", "policy", "edit"]
ENV = {u: U.taxonomy_envelope(u, pool) for u in UPD}
MASS_GRID = [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
EPS = [0.005, 0.01, 0.02, 0.05]
ALPHA = 0.05
S = {}


def ld(model, tag):
    p = f"{RES}/{model}/{tag}.npz"
    return fp.load_eval(p) if os.path.exists(p) else None


def flips(a, b):
    return a["decision"] != b["decision"]


def jacc(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def decisive_tau(base, null, target=0.001):
    """Smallest margin threshold such that the self-disagreement (null) flip rate among decisions with incumbent margin
    above it is below target; chosen on the anchor split only."""
    d = flips(base, null) & ANC
    for tau in np.arange(0.0, 3.01, 0.05):
        keep = ANC & (base["margin"] > tau)
        if keep.sum() and d[keep].mean() < target:
            return float(tau)
    return 3.0


def footprint_stats(base, new, E, tau, mask=CERT):
    D = flips(base, new)
    out = {}
    m = mask
    out["footprint_mass"] = float(D[m].mean())
    out["envelope_mass"] = float(E[m].mean())
    out["in_change_rate"] = float(D[m & E].mean()) if (m & E).any() else float("nan")
    out["leakage"] = float(D[m & ~E].mean())
    out["leaked_mass"] = float((D & ~E)[m].mean())
    out["containment"] = float((D & E)[m].sum() / max(1, D[m].sum()))
    dec = base["margin"] > tau
    out["leakage_decisive"] = float(D[m & ~E & dec].mean())
    out["decisive_frac_out"] = float(dec[m & ~E].mean())
    corr0 = base["decision"] == ans; corr1 = new["decision"] == ans
    o = m & ~E & (ans >= 0)
    out["out_acc_change"] = float(corr1[o].mean() - corr0[o].mean())
    out["out_neg_flip"] = float((corr0 & ~corr1)[o].mean())
    out["out_pos_flip"] = float((~corr0 & corr1)[o].mean())
    out["per_domain_flip"] = {d: float(D[m & (dom == d)].mean()) for d in sorted(set(dom))}
    return out, D


def curve(score, D, mask_fit, mask_eval, grid=MASS_GRID):
    """Declared envelope = top-m by score, threshold fixed on the fit (anchor) split; leakage on eval split."""
    res = []
    for mm in grid:
        thr = np.quantile(score[mask_fit], 1 - mm)
        E = score >= thr
        mev = mask_eval
        res.append(dict(mass=float(E[mev].mean()), leakage=float(D[mev & ~E].mean()) if (mev & ~E).any() else 0.0,
                        containment=float((D & E)[mev].sum() / max(1, D[mev].sum()))))
    return res


def auroc(score, D, mask):
    y = D[mask]
    return float(roc_auc_score(y, score[mask])) if 0 < y.sum() < len(y) else float("nan")


def knn_sim(emb_pool, emb_train):
    import torch
    a = torch.tensor(emb_pool, dtype=torch.float32, device="cuda")
    b = torch.tensor(emb_train, dtype=torch.float32, device="cuda")
    out = []
    for i in range(0, len(a), 4096):
        s = a[i:i + 4096] @ b.T
        out.append(s.topk(5, dim=1).values.mean(1).cpu().numpy())
    return np.concatenate(out)


def certificate_row(D, E, rng_seed=0, dec=None, alpha=None, n_perm=20):
    ALPHA = alpha if alpha is not None else globals()["ALPHA"]
    rng = np.random.RandomState(rng_seed)
    idx = np.where(CERT)[0]; rng.shuffle(idx)
    z_uncond = (D & ~E)[idx]
    idx_out = idx[~E[idx]]
    z_cond = D[idx_out]
    row = {}
    for eps in EPS:
        ok, n, u = C.certify_stream(z_uncond, eps, ALPHA)
        row[f"uncond_eps{eps}"] = dict(certified=ok, n=n, ucb=u)
        ok, n, u = C.certify_stream(z_cond, eps, ALPHA)
        row[f"cond_eps{eps}"] = dict(certified=ok, n=n, ucb=u)
    if dec is not None:
        idx_dec = idx[(~E[idx]) & dec[idx]]
        z_dec = D[idx_dec]
        for eps in EPS:
            ok, n, u = C.certify_stream(z_dec, eps, ALPHA)
            row[f"dec_eps{eps}"] = dict(certified=ok, n=n, ucb=u)
        row["ucb_full_dec"] = C.ucb_mixture(int(z_dec.sum()), len(z_dec), ALPHA)
    row["ucb_full_uncond"] = C.ucb_mixture(int(z_uncond.sum()), len(z_uncond), ALPHA)
    # distribution of samples-to-certify over random audit orders (conditional form)
    out_ids = np.where(CERT & ~E)[0]
    for eps in EPS:
        ns = []
        for k in range(n_perm):
            o = np.random.RandomState(100 + k).permutation(out_ids)
            ok, n, _ = C.certify_stream(D[o], eps, ALPHA)
            ns.append(n if ok else np.nan)
        ns = np.array(ns, float)
        row[f"perm_cond_eps{eps}"] = dict(frac_certified=float(np.mean(~np.isnan(ns))),
                                          median_n=float(np.nanmedian(ns)) if (~np.isnan(ns)).any() else None,
                                          q25=float(np.nanpercentile(ns, 25)) if (~np.isnan(ns)).any() else None,
                                          q75=float(np.nanpercentile(ns, 75)) if (~np.isnan(ns)).any() else None)
    # one-sided mixture-of-SPRTs test at each pre-declared eps (same audit orders)
    for eps in EPS:
        ns = []
        for k in range(n_perm):
            o = np.random.RandomState(100 + k).permutation(out_ids)
            ok, n = C.certify_onesided(D[o], eps, ALPHA / len(EPS), check_every=5)  # eps not in the commitment: Bonferroni over the eps grid
            ns.append(n if ok else np.nan)
        ns = np.array(ns, float)
        row[f"perm_os_eps{eps}"] = dict(frac_certified=float(np.mean(~np.isnan(ns))),
                                        median_n=float(np.nanmedian(ns)) if (~np.isnan(ns)).any() else None)
    row["alpha"] = ALPHA
    row["ucb_full_cond"] = C.ucb_mixture(int(z_cond.sum()), len(z_cond), ALPHA)
    return row


def main():
    judge = np.load(f"{RES}/judge_qwen7b.npz") if os.path.exists(f"{RES}/judge_qwen7b.npz") else None
    for model in ["qwen7b", "llama8b", "qwen32b", "mistral7b", "qwen72b", "qwen1.5b"]:
        base, null = ld(model, "base"), ld(model, "null_order1")
        if base is None:
            continue
        SM = S.setdefault(model, {})
        tau = decisive_tau(base, null) if null is not None else 0.5
        nf = flips(base, null) if null is not None else np.zeros(N, bool)
        SM["null"] = dict(flip_rate=float(nf[CERT].mean()), tau=tau,
                          flip_rate_decisive=float(nf[CERT & (base["margin"] > tau)].mean()),
                          decisive_frac=float((base["margin"] > tau)[CERT].mean()),
                          base_acc=float((base["decision"] == ans)[ans >= 0].mean()))
        # margin law: flip prob vs base margin bins, pooled over controlled updates
        bins = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 100]
        tr = np.load(f"{RES}/{model}/train_emb.npz") if os.path.exists(f"{RES}/{model}/train_emb.npz") else None
        SM["updates"] = {}
        for u in UPD:
            E = ENV[u]
            runs = {s: ld(model, f"{u}_s{s}") for s in (0, 1)}
            if runs[0] is None:
                continue
            st, D0 = footprint_stats(base, runs[0], E, tau)
            st["in_effect"] = U.in_envelope_effect(u, pool, base, runs[0], mask=CERT)
            ent = dict(seed0=st)
            if runs[1] is not None:
                st1, D1 = footprint_stats(base, runs[1], E, tau)
                st1["in_effect"] = U.in_envelope_effect(u, pool, base, runs[1], mask=CERT)
                ent["seed1"] = st1
                o = CERT & ~E
                p0, p1 = D0[o].mean(), D1[o].mean()
                ent["seed_jaccard_out"] = jacc(D0 & o, D1 & o)
                ent["seed_jaccard_chance"] = float(p0 * p1 / (p0 + p1 - p0 * p1))
                ent["seed_jaccard_out_decisive"] = jacc(D0 & o & (base["margin"] > tau), D1 & o & (base["margin"] > tau))
            o = CERT & ~E & D0
            R_, a_ = U.train_set(u)
            ent["flip_new_letter_dist"] = (np.bincount(runs[0]["decision"][o], minlength=5)[:5] / max(1, o.sum())).tolist()
            ent["train_answer_dist"] = (np.bincount(a_, minlength=5)[:5] / len(a_)).tolist()
            ent["base_letter_dist_out"] = (np.bincount(base["decision"][CERT & ~E], minlength=5)[:5] / max(1, (CERT & ~E).sum())).tolist()
            mb = np.digitize(base["margin"], bins) - 1
            ent["margin_law"] = [dict(lo=bins[k], hi=bins[k + 1], n=int((CERT & ~E & (mb == k)).sum()),
                                      flip=float(D0[CERT & ~E & (mb == k)].mean()) if (CERT & ~E & (mb == k)).any() else None,
                                      null=float(nf[CERT & (mb == k)].mean()) if (CERT & (mb == k)).any() else None)
                                 for k in range(len(bins) - 1)]
            # declarations
            scores = {"fragility": -base["margin"].astype(np.float64)}
            if judge is not None:
                scores["nl_judge"] = judge[u].astype(np.float64)
            if tr is not None:
                scores["knn_data"] = knn_sim(base["emb"], tr[u]).astype(np.float64)
            scores["random"] = np.random.RandomState(0).rand(N)
            scores["oracle"] = D0.astype(np.float64) + 1e-3 * np.random.RandomState(1).rand(N)
            pilot = ld(model, f"pilot_{u}")
            feats = [base["margin"][:, None], np.log1p(base["margin"])[:, None], base["probs"][:, :5].astype(np.float64),
                     np.eye(5)[np.clip(base["decision"], 0, 4)]]
            if "knn_data" in scores:
                feats.append(scores["knn_data"][:, None])
            if "nl_judge" in scores:
                feats.append(scores["nl_judge"][:, None])
            from sklearn.decomposition import PCA
            pca = PCA(32, random_state=0).fit(base["emb"][ANC].astype(np.float32))
            feats.append(pca.transform(base["emb"].astype(np.float32)))
            X = np.concatenate(feats, 1)
            X = (X - X[ANC].mean(0)) / (X[ANC].std(0) + 1e-6)
            if pilot is not None:
                Dp = flips(base, pilot)
                clf = LogisticRegression(max_iter=2000, C=1.0).fit(X[ANC], Dp[ANC])
                scores["pilot_forecast"] = clf.predict_proba(X)[:, 1]
                ent["pilot_footprint_mass"] = float(Dp[ANC].mean())
                ent["pilot_vs_full_jaccard_anchor"] = jacc(Dp & ANC, D0 & ANC)
            # post-hoc learnability upper bound: same features, trained on the real update's anchor flips
            clf2 = LogisticRegression(max_iter=2000, C=1.0).fit(X[ANC], D0[ANC])
            scores["posthoc_learned"] = clf2.predict_proba(X)[:, 1]
            if "nl_judge" in scores:
                from scipy.stats import rankdata
                scores["fragility+judge"] = rankdata(scores["fragility"]) + rankdata(scores["nl_judge"])
            ent["declarations"] = {}
            for k, sc in scores.items():
                ent["declarations"][k] = dict(auroc=auroc(sc, D0, CERT), auroc_out=auroc(sc, D0, CERT & ~E),
                                              curve=curve(sc, D0, ANC, CERT))
            # combined: union of taxonomy scope with top-m by forecast (a practical declaration)
            ent["certificate_taxonomy"] = certificate_row(D0, E)
            # hard-gating reference: route to the update only when the NL judge says in-scope (mass matched to taxonomy)
            if "nl_judge" in scores:
                thr = np.quantile(scores["nl_judge"][ANC], 1 - E[ANC].mean())
                G = scores["nl_judge"] >= thr
                gated = dict(decision=np.where(G, runs[0]["decision"], base["decision"]))
                if "evaluated" in runs[0]:
                    gated["evaluated"] = runs[0]["evaluated"]
                Dg = flips(base, gated)
                ent["gating_nl"] = dict(leakage=float(Dg[CERT & ~E].mean()), in_effect=U.in_envelope_effect(u, pool, base, gated, mask=CERT),
                                        judge_recall=float(G[CERT & E].mean()), judge_fpr=float(G[CERT & ~E].mean()))
            SM["updates"][u] = ent
        # localization variants
        SM["variants"] = {}
        for p in sorted(glob.glob(f"{RES}/{model}/*.npz")):
            tag = os.path.basename(p)[:-4]
            u = tag.split("_")[0]
            if u not in UPD or tag.startswith("pilot") or tag.endswith("_s0") or tag.endswith("_s1"):
                continue
            new = ld(model, tag)
            st, D = footprint_stats(base, new, ENV[u], tau)
            st["in_effect"] = U.in_envelope_effect(u, pool, base, new, mask=CERT)
            st["certificate"] = certificate_row(D, ENV[u], dec=base["margin"] > tau, alpha=ALPHA / 4)
            HO = np.isin(dom, ["stem", "humanities_social", "commonsense", "science_qa"])
            st["leakage_heldout_domains"] = float(D[CERT & ~ENV[u] & HO].mean())
            st["leakage_anchored_domains"] = float(D[CERT & ~ENV[u] & ~HO].mean())
            SM["variants"][tag] = st
        for u in UPD:
            b0 = ld(model, f"{u}_s0")
            if b0 is not None:
                st, D = footprint_stats(base, b0, ENV[u], tau)
                st["in_effect"] = U.in_envelope_effect(u, pool, base, b0, mask=CERT)
                st["certificate"] = certificate_row(D, ENV[u], dec=base["margin"] > tau, alpha=ALPHA / 4)
                HO = np.isin(dom, ["stem", "humanities_social", "commonsense", "science_qa"])
                st["leakage_heldout_domains"] = float(D[CERT & ~ENV[u] & HO].mean())
                st["leakage_anchored_domains"] = float(D[CERT & ~ENV[u] & ~HO].mean())
                SM["variants"][f"{u}_base"] = st
        # sequential composition
        SM["chains"] = {}
        for ch, first, last in [("plain", "med_s0", "4"), ("kl", "chain_kl_1", "4"), ("plain_mix", "med_s0", "4mix"), ("kl_mix", "chain_kl_1", "4mix")]:
            cb = ch.split("_")[0]
            tags = [first] + [f"chain_{cb}_{k}" for k in ("2", "3", last)]
            evs = [ld(model, t) for t in tags]
            if any(e is None for e in evs):
                continue
            seq = [base] + evs
            union = np.zeros(N, bool)
            steps = []
            bound = 0.0
            for k, u in enumerate(UPD):
                union |= ENV[u]
                Dk = flips(seq[k], seq[k + 1])
                lm = float((Dk & ~ENV[u])[CERT].mean())
                bound += lm
                cum = flips(seq[0], seq[k + 1])
                steps.append(dict(update=u, step_leaked_mass=lm, bound=bound,
                                  observed_out_union=float((cum & ~union)[CERT].mean()),
                                  observed_out_union_decisive=float((cum & ~union & (base["margin"] > tau))[CERT].mean()),
                                  in_effect=U.in_envelope_effect(u, pool, seq[k], seq[k + 1], mask=CERT)))
            # certified version with lifetime error spending
            cert_bound = 0.0
            for k, u in enumerate(UPD):
                Dk = flips(seq[k], seq[k + 1])
                idx = np.where(CERT)[0]
                z = (Dk & ~ENV[u])[idx]
                cert_bound += C.ucb_mixture(int(z.sum()), len(z), C.spend(ALPHA, k + 1))
                steps[k]["certified_bound"] = cert_bound
            SM["chains"][ch] = steps
        # public fine-tunes: per-domain flip rates relative to base
        SM["public"] = {}
        for p in sorted(glob.glob(f"{RES}/{model}/pub_*.npz")):
            tag = os.path.basename(p)[:-4]
            new = ld(model, tag)
            D = flips(base, new)
            corr0 = base["decision"] == ans; corr1 = new["decision"] == ans
            SM["public"][tag] = dict(flip=float(D[CERT].mean()),
                                     flip_decisive=float(D[CERT & (base["margin"] > tau)].mean()),
                                     acc_base=float(corr0[CERT].mean()), acc_new=float(corr1[CERT].mean()),
                                     neg_flip=float((corr0 & ~corr1)[CERT].mean()), pos_flip=float((~corr0 & corr1)[CERT].mean()),
                                     per_domain_flip={d: float(D[CERT & (dom == d)].mean()) for d in sorted(set(dom))},
                                     per_domain_acc_change={d: float(corr1[CERT & (dom == d)].mean() - corr0[CERT & (dom == d)].mean()) for d in sorted(set(dom))},
                                     certificate_all_out=C.ucb_mixture(int(D[CERT].sum()), int(CERT.sum()), ALPHA))
    json.dump(S, open(f"{RES}/summary.json", "w"), indent=1)
    print("wrote summary")


if __name__ == "__main__":
    main()
