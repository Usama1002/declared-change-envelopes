#!/usr/bin/env bash
# Other incumbents: Qwen2.5-1.5B (weak incumbent), Mistral-7B-v0.3 (third family, including the diagnostics for its
# anchoring plateau), Qwen2.5-32B (one H200), Qwen2.5-72B (two H200s: export CUDA_VISIBLE_DEVICES=0,1).
# The 1.5B and Mistral incumbents are evaluated in 01_base_and_floors.sh.
source "$(dirname "$0")/env.sh"

# Qwen2.5-1.5B-Instruct
for u in med legal policy; do
  upd qwen1.5b ${u}_s0 --update $u
  upd qwen1.5b ${u}_kl1 --update $u --kl_lambda 1
  upd qwen1.5b ${u}_kl10 --update $u --kl_lambda 10
done

# Mistral-7B-Instruct-v0.3
for u in med policy; do
  upd mistral7b ${u}_s0 --update $u
  upd mistral7b ${u}_kl1 --update $u --kl_lambda 1
  upd mistral7b ${u}_kl10 --update $u --kl_lambda 10
  upd mistral7b ${u}_kl1a64 --update $u --kl_lambda 1 --anchor_bs 64
  upd mistral7b ${u}_kl1let --update $u --kl_lambda 1 --kl_mode letters
  upd mistral7b ${u}_lr3e-5 --update $u --lr 3e-5
  upd mistral7b ${u}_kl1lr3e-5 --update $u --lr 3e-5 --kl_lambda 1
done

# Qwen2.5-32B-Instruct and Qwen2.5-72B-Instruct (72B is split layer-wise over all visible GPUs)
for m in qwen32b qwen72b; do
  run "$PY" src/run_eval.py --model $m --family $m --tag base --emb --bs 16
  run "$PY" src/run_eval.py --model $m --family $m --tag null_order1 --bs 16 --order_seed 1
  for u in med policy; do
    EVAL_BS=16 upd $m ${u}_s0 --update $u
    EVAL_BS=16 upd $m ${u}_kl1 --update $u --kl_lambda 1
    EVAL_BS=16 upd $m ${u}_s1 --update $u --seed 1
    EVAL_BS=16 upd $m ${u}_kl1s1 --update $u --seed 1 --kl_lambda 1
  done
done
