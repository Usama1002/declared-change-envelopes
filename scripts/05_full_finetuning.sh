#!/usr/bin/env bash
# Full fine-tuning of every weight (fp32 master weights, bf16 autocast, 8-bit AdamW, gradient checkpointing; the
# anchored runs keep a frozen bf16 copy of the incumbent as reference), med and policy, Qwen2.5-7B and Llama-3.1-8B.
# --anchor_bs = anchor prompts per step (default: the training batch size, 16).
source "$(dirname "$0")/env.sh"

for m in qwen7b llama8b; do
  for u in med policy; do
    upd $m ${u}_fft --update $u --full --lr 1e-5
    upd $m ${u}_fftlr3e-6 --update $u --full --lr 3e-6
    upd $m ${u}_fftlr1e-6 --update $u --full --lr 1e-6
    upd $m ${u}_fftkl1 --update $u --full --lr 1e-5 --kl_lambda 1
    upd $m ${u}_fftkl10 --update $u --full --lr 1e-5 --kl_lambda 10
    upd $m ${u}_fftkl1lr3e-6 --update $u --full --lr 3e-6 --kl_lambda 1
    upd $m ${u}_fftkl10lr3e-6 --update $u --full --lr 3e-6 --kl_lambda 10
    upd $m ${u}_fftkl10ab64lr3e-6 --update $u --full --lr 3e-6 --kl_lambda 10 --anchor_bs 64
    upd $m ${u}_fftkl10ab64lr1e-6 --update $u --full --lr 1e-6 --kl_lambda 10 --anchor_bs 64
    upd $m ${u}_fftkl10ab128lr3e-6 --update $u --full --lr 3e-6 --kl_lambda 10 --anchor_bs 128
    upd $m ${u}_fftlr3e-6s1 --update $u --full --lr 3e-6 --seed 1
    upd $m ${u}_fftkl10ab64lr3e-6s1 --update $u --full --lr 3e-6 --kl_lambda 10 --anchor_bs 64 --seed 1
  done
done
