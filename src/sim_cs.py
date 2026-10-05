"""Simulation: validity and sample cost of anytime-valid leakage certificates vs naive monitored Clopper-Pearson."""
import json, numpy as np
from scipy.special import betaln
from scipy.stats import beta as B
from config import RESULTS
import certify as C

rng = np.random.default_rng(0)
ALPHA, EPS, T, EVERY, REPS = 0.05, 0.02, 20000, 5, 2000
grid = np.arange(EVERY, T + 1, EVERY)


def mix_ucb_vec(S, t, alpha=ALPHA):
    lo = S / t; hi = np.full_like(lo, 1 - 1e-12)
    thr = np.log(1 / alpha)
    for _ in range(50):
        mid = (lo + hi) / 2
        lm = betaln(1 + S, 1 + t - S) - betaln(1, 1) - S * np.log(mid) - (t - S) * np.log1p(-mid)
        ok = lm < thr
        lo = np.where(ok, mid, lo); hi = np.where(ok, hi, mid)
    return hi


def run(lam):
    Z = rng.random((REPS, T)) < lam
    Sg = np.cumsum(Z, 1)[:, grid - 1].astype(float)
    tg = np.broadcast_to(grid, Sg.shape).astype(float)
    u_mix = mix_ucb_vec(Sg, tg)
    u_cp = np.where(Sg >= tg, 1.0, B.ppf(1 - ALPHA, Sg + 1, tg - Sg))
    out = {}
    for name, u in [("mixture_cs", u_mix), ("cp_monitored", u_cp)]:
        cert = u <= EPS
        any_cert = cert.any(1)
        first = np.where(any_cert, grid[np.argmax(cert, 1)], np.nan)
        out[name] = dict(cert_rate=float(any_cert.mean()), median_n=float(np.nanmedian(first)) if any_cert.any() else None)
    # one-sided mixture-of-SPRTs test at the declared eps (Proposition 3), same monitoring grid
    q = EPS * C.OS_Q
    lp = np.log(np.maximum(q, 1e-300) / EPS); lq = np.log1p(-q) - np.log1p(-EPS)
    lc = Sg[..., None] * lp + (tg - Sg)[..., None] * lq
    lc = np.where((q == 0) & (Sg[..., None] > 0), -np.inf, lc)
    m = lc.max(-1)
    L = m + np.log((C.OS_W * np.exp(lc - m[..., None])).sum(-1))
    cert = L >= np.log(1 / ALPHA)
    any_cert = cert.any(1)
    first = np.where(any_cert, grid[np.argmax(cert, 1)], np.nan)
    out["onesided_test"] = dict(cert_rate=float(any_cert.mean()), median_n=float(np.nanmedian(first)) if any_cert.any() else None)
    # fixed-n CP at the final sample (valid, no monitoring)
    out["cp_fixed_final"] = dict(cert_rate=float((u_cp[:, -1] <= EPS).mean()))
    return out


res = {}
for lam in [0.0, 0.005, 0.01, 0.015, 0.02, 0.025]:
    res[str(lam)] = run(lam)
    print(lam, res[str(lam)], flush=True)
json.dump(dict(alpha=ALPHA, eps=EPS, T=T, every=EVERY, reps=REPS, results=res), open(f"{RESULTS}/sim_cs.json", "w"), indent=1)
