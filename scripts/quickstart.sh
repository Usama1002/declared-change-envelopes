#!/usr/bin/env bash
# One update end to end on Qwen2.5-1.5B-Instruct (about 10 minutes on one H200 after the data are built):
# incumbent decisions, the medical update with plain LoRA and with envelope anchoring (beta 1), then effect,
# leakage, and the anytime-valid certificate. Requires 00_data.sh. Tags start with "quick_" so that they are ignored
# by the analysis scripts. Set MAX_STEPS (e.g. MAX_STEPS=2) for a smoke test.
source "$(dirname "$0")/env.sh"

STEPS=()
[ -n "${MAX_STEPS:-}" ] && STEPS=(--max_steps "$MAX_STEPS")
[ -f "$RES/qwen1.5b/base.npz" ] || run "$PY" src/run_eval.py --model qwen1.5b --family qwen1.5b --tag base --emb
upd qwen1.5b quick_med_plain --update med "${STEPS[@]}"
upd qwen1.5b quick_med_anchored --update med --kl_lambda 1 "${STEPS[@]}"

PYTHONPATH=src "$PY" - <<'PY'
import numpy as np
import analyze as A, certify as C, updates as U
base = A.ld("qwen1.5b", "base")
E = A.ENV["med"]
for tag in ["quick_med_plain", "quick_med_anchored"]:
    new = A.ld("qwen1.5b", tag)
    st, D = A.footprint_stats(base, new, E, tau=0.5)
    effect = U.in_envelope_effect("med", A.pool, base, new, mask=A.CERT)
    # audit stream: out-of-envelope audit items in one random order fixed in advance
    z = D[np.random.RandomState(7).permutation(np.where(A.CERT & ~E)[0])]
    cert = [eps for eps in (0.01, 0.02, 0.05, 0.10) if C.certify_stream(z, eps, 0.05)[0]]
    print(f"{tag:20s} in-scope effect {100 * effect:+5.1f} pp | leakage {100 * st['leakage']:5.2f}% "
          f"| 95% anytime-valid upper bound {100 * C.ucb_mixture(int(z.sum()), len(z), 0.05):5.2f}% "
          f"| smallest certified eps: {f'{100 * cert[0]:g}%' if cert else 'none'}")
PY
