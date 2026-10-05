"""Validity of the mixture confidence sequence when the audit stream is a random ordering of a finite pool
(sampling without replacement), as in our experiments. Miscoverage = P(exists t: lambda_pool > U_t)
= P(exists t: log mixture statistic at lambda_pool >= log(1/alpha)). Writes results/sim_wor.json."""
import json, numpy as np
from scipy.special import betaln
from config import RESULTS

rng = np.random.default_rng(0)
out = []
for N in [2000, 8000]:
    for lam in [0.01, 0.02, 0.05, 0.10]:
        for alpha in [0.05, 0.0125]:
            k = int(round(lam * N)); lam_pool = k / N
            R = 20000; miss = 0
            t = np.arange(1, N + 1)
            for r0 in range(0, R, 500):
                Z = np.zeros((500, N), np.int8); Z[:, :k] = 1
                Z = rng.permuted(Z, axis=1)
                S = Z.cumsum(1)
                lm = betaln(1 + S, 1 + t - S) - betaln(1, 1) - (S * np.log(lam_pool) + (t - S) * np.log1p(-lam_pool))
                miss += int((lm.max(1) >= np.log(1 / alpha)).sum())
            out.append(dict(N=N, lam=lam_pool, alpha=alpha, miscoverage=miss / R, orders=R))
            print(out[-1], flush=True)
json.dump(out, open(f"{RESULTS}/sim_wor.json", "w"), indent=1)
