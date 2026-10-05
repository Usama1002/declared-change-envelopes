#!/usr/bin/env bash
# Held-out (confirmatory) audits on the half of the audit split that was never evaluated during development
# (--eval_on complement): the plain adapters of 02, anchored candidates of every update type trained anew, and
# written-answer certificates (greedy generation, 3,000 out-of-scope confirmatory queries). The seed replicates
# rep_* of 02 are evaluated on the full pool and enter the confirmatory summary as well. Requires 02.
source "$(dirname "$0")/env.sh"

# plain LoRA adapters of 02, evaluated on the confirmatory half
for m in qwen7b llama8b; do
  for u in med legal policy edit; do
    for s in 0 1; do
      upd $m comp_${u}_s$s --update $u --init $CKPT/${m}_${u}_s$s --eval_only --eval_on complement
    done
  done
done

# anchored candidates trained anew and evaluated only on the confirmatory half
for m in qwen7b llama8b; do
  for u in med policy; do
    upd $m cf_${u}_fftkl10ab64lr1e-6 --update $u --full --lr 1e-6 --kl_lambda 10 --anchor_bs 64 --eval_on complement
    upd $m cf_${u}_grpokl1 --update $u --objective grpo --kl_lambda 1 --eval_on complement
  done
done
for u in med legal policy; do upd qwen1.5b cf_${u}_kl10 --update $u --kl_lambda 10 --eval_on complement; done
# Qwen2.5-72B: two GPUs
for u in med policy; do EVAL_BS=16 upd qwen72b cf_${u}_kl1 --update $u --kl_lambda 1 --eval_on complement; done

# written-answer certificates on the confirmatory half (the med runs also measure the generation floor)
for m in qwen7b llama8b; do
  run "$PY" src/gen_check.py --model $m --split confirm --n 3000 --scope_update med --adapters med_plainsv,med_anchsv --floor
  run "$PY" src/gen_check.py --model $m --split confirm --n 3000 --scope_update legal --adapters legal_plainsv,legal_anchsv
  run "$PY" src/gen_check.py --model $m --split confirm --n 3000 --scope_update policy --adapters policy_plainsv,policy_anchsv
done
