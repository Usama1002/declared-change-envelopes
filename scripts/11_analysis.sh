#!/usr/bin/env bash
# Metrics, declarations, certificates, and summaries from the per-run decision files. analyze.py uses one GPU for the
# kNN declaration; everything else runs on CPU. Each script overwrites the corresponding JSON file in results/.
source "$(dirname "$0")/env.sh"

run "$PY" src/sim_cs.py                 # sim_cs.json: validity and sample cost of the certificates (simulation)
run "$PY" src/sim_wor.py                # sim_wor.json: validity under sampling without replacement (simulation)
run "$PY" src/analyze.py                # summary.json: footprints, declarations, certificates, chains, public fine-tunes
run "$PY" src/analyze_extra.py          # extra.json: bootstrap CIs, routing, judge-envelope anchoring, editors, chain order 2, TOST
run "$PY" src/analyze_confirm.py        # confirm.json: confirmatory half, seed replicates, lr x epoch sweep
run "$PY" src/analyze_update_types.py   # update_types.json: full fine-tuning, GRPO, RLVR, 1.5B and 72B
run "$PY" src/analyze_shift.py          # shift_confirm.json: anchor/audit shift, focal distillation, held-out certificates
run "$PY" src/kl_leak.py                # kl_leak.json: leakage vs out-of-scope letter KL for every variant
