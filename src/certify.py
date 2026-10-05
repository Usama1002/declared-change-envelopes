"""Label-free, anytime-valid footprint certificates.

Z_t in {0,1} is the leak indicator on the t-th unlabeled traffic sample: Z = 1[d_old(x) != d_new(x) and c(x) = 0]
(unconditional leaked mass) or, restricted to the c(x)=0 substream, the conditional leakage. Under i.i.d. traffic,
M_t(p) = B(a+S_t, b+t-S_t)/B(a,b) / (p^S_t (1-p)^(t-S_t)) is a nonnegative martingale with M_0 = 1 at the true p
(Beta-mixture of likelihood ratios; Robbins 1970). Ville's inequality gives P(exists t: M_t(p*) >= 1/alpha) <= alpha,
so U_t = sup{p : M_t(p) < 1/alpha} is an anytime-valid upper confidence sequence.
"""
import math
import numpy as np
from scipy.special import betaln
from scipy.stats import beta as beta_dist


def log_mix(S, t, p, a=1.0, b=1.0):
    return betaln(a + S, b + t - S) - betaln(a, b) - S * np.log(p) - (t - S) * np.log1p(-p)


def ucb_mixture(S, t, alpha, a=1.0, b=1.0, iters=60):
    """Upper end of the anytime-valid confidence sequence after t samples with S successes."""
    if t == 0:
        return 1.0
    lo, hi = S / t, 1.0 - 1e-12
    thr = math.log(1 / alpha)
    if log_mix(S, t, hi, a, b) < thr:
        return 1.0
    for _ in range(iters):
        mid = (lo + hi) / 2
        if log_mix(S, t, mid, a, b) < thr:
            lo = mid
        else:
            hi = mid
    return hi


def ucb_clopper_pearson(S, t, alpha):
    """Fixed-n one-sided Clopper-Pearson upper bound (NOT valid under continuous monitoring)."""
    if S >= t:
        return 1.0
    return float(beta_dist.ppf(1 - alpha, S + 1, t - S))


def certify_stream(z, eps, alpha, check_every=25, method="mixture"):
    """Return (certified: bool, samples_used: int, final_ucb: float). Stops at the first t with U_t <= eps."""
    S = 0
    f = ucb_mixture if method == "mixture" else ucb_clopper_pearson
    for t in range(1, len(z) + 1):
        S += int(z[t - 1])
        if t % check_every == 0 or t == len(z):
            u = f(S, t, alpha)
            if u <= eps:
                return True, t, u
    return False, len(z), f(S, len(z), alpha)


def ucb_path(z, alpha, grid):
    cs = np.cumsum(z)
    return np.array([ucb_mixture(int(cs[t - 1]), t, alpha) for t in grid])


def spend(alpha, t):
    """Lifetime error spending: alpha_t = alpha * 6 / (pi^2 t^2); sum over t >= 1 equals alpha."""
    return alpha * 6.0 / (math.pi ** 2 * t ** 2)


def samples_needed_zero_leak(eps, alpha):
    """Smallest t with ucb_mixture(0, t, alpha) <= eps (S=0 case): i.e. (t+1)(1-eps)^t <= alpha."""
    t = 1
    while (t + 1) * (1 - eps) ** t > alpha:  # M_t(eps)=1/((t+1)(1-eps)^t) >= 1/alpha
        t += 1
    return t


# ---------------------------------------------------------------- one-sided test for a pre-declared tolerance
OS_Q = np.array([0.0] + [k / 10 for k in range(1, 10)])        # alternatives q = eps * OS_Q, all below eps
OS_W = np.array([0.5] + [0.5 / 9] * 9)                          # pre-registered prior weights


def log_os(S, t, eps):
    """log of the mixture-of-SPRTs e-process for H0: lambda >= eps.
    Each component prod (q/eps)^Z ((1-q)/(1-eps))^(1-Z) with q < eps has per-step expectation
    lambda q/eps + (1-lambda)(1-q)/(1-eps) <= 1 for every lambda >= eps, so it (and any fixed mixture) is a nonnegative
    supermartingale under H0 and Ville's inequality makes 'certify when it reaches 1/alpha' valid under continuous monitoring."""
    q = eps * OS_Q
    with np.errstate(divide="ignore"):
        lc = np.where(S > 0, S * np.log(np.maximum(q, 1e-300) / eps), 0.0) + (t - S) * (np.log1p(-q) - np.log1p(-eps))
    lc = np.where((q == 0) & (S > 0), -np.inf, lc)
    m = lc.max()
    return m + np.log(np.sum(OS_W * np.exp(lc - m)))


def certify_onesided(z, eps, alpha, check_every=1):
    """Return (certified, samples_used). Stops at the first t where the e-process reaches 1/alpha (vectorized)."""
    z = np.asarray(z, float)
    if len(z) == 0:
        return False, 0
    S = np.cumsum(z); t = np.arange(1, len(z) + 1)
    q = eps * OS_Q
    lp = np.log(np.maximum(q, 1e-300) / eps)
    lc = S[:, None] * lp[None, :] + (t - S)[:, None] * (np.log1p(-q) - np.log1p(-eps))[None, :]
    lc[:, q == 0] = np.where(S[:, None] > 0, -np.inf, lc[:, q == 0])
    m = lc.max(1)
    L = m + np.log((OS_W[None, :] * np.exp(lc - m[:, None])).sum(1))
    hit = np.where((L >= math.log(1 / alpha)) & (t % check_every == 0))[0]
    return (True, int(hit[0] + 1)) if len(hit) else (False, len(z))
