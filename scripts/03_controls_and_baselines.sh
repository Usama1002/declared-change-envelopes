#!/usr/bin/env bash
# Alternatives to anchoring and ablations of it, for med and policy on Qwen2.5-7B and Llama-3.1-8B: plain LoRA
# learning-rate x epoch grid, adapter scaling, L2 to the incumbent, incumbent replay, KL on generic chat text, KL on
# generic multiple choice, KL on all traffic, anchoring on the NL-judge envelope, anchor/audit domain shift, focal
# distillation, letter-only KL, rank and learning-rate variants. Requires 01 and 02 (adapters med_s0, policy_s0).
source "$(dirname "$0")/env.sh"

HO=stem,humanities_social,commonsense,science_qa                                             # held-out domain group
NONHO=legal,news_topic,entity_facts_edited,entity_facts_neighbor,reading,sentiment,medical   # its complement

# generic chat anchors: 3,000 Alpaca instructions with the incumbent's greedy responses (data/anchors_generic_<model>.jsonl)
for m in qwen7b llama8b; do run "$PY" src/build_generic_anchors.py --model $m; done

for m in qwen7b llama8b; do
  for u in med policy; do
    # plain LoRA grid: lr in {1e-5, 3e-5, 1e-4} x epochs in {1, 2, 4} (lr 1e-4 x 2 epochs is ${u}_s0; on Qwen,
    # lr 3e-5 x 2 epochs is ${u}_lr3e-5 below)
    for lr in 1e-5 3e-5; do
      for ep in 1 2 4; do
        [ $m = qwen7b ] && [ $lr = 3e-5 ] && [ $ep = 2 ] && continue
        upd $m sweep_${u}_lr${lr}_e$ep --update $u --lr $lr --epochs $ep
      done
    done
    upd $m sweep_${u}_lr1e-4_e1 --update $u --epochs 1
    upd $m sweep_${u}_lr1e-4_e4 --update $u --epochs 4
    # adapter scaling (weight interpolation toward the incumbent) of the seed-0 plain adapter
    for sc in 0.25 0.5 0.75; do
      upd $m ${u}_scale$sc --update $u --init $CKPT/${m}_${u}_s0 --init_scale $sc --eval_only
    done
    # L2 penalty on the LoRA weight update
    upd $m ${u}_l2d0.1 --update $u --l2_delta 0.1
    upd $m ${u}_l2d1 --update $u --l2_delta 1
    # KL on generic chat text, on generic multiple choice from outside the pool, and on all anchor traffic incl. the envelope
    upd $m ${u}_genkl1 --update $u --kl_lambda 1 --generic_anchor
    upd $m ${u}_mcgenkl1 --update $u --kl_lambda 1 --anchor_file anchors_mc_generic.jsonl
    upd $m ${u}_klall --update $u --kl_lambda 1 --anchor_all
    upd $m ${u}_klall10 --update $u --kl_lambda 10 --anchor_all
    # anchoring with the NL-judge envelope as the declaration
    upd $m judgeenv_${u}_kl1 --update $u --kl_lambda 1 --envelope judge
    # anchor/audit domain shift: anchors exclude one domain group, the audit is on that group
    upd $m ${u}_kl1ho --update $u --kl_lambda 1 --anchor_exclude $HO
    upd $m ${u}_kl10ho --update $u --kl_lambda 10 --anchor_exclude $HO
    upd $m ${u}_kl1hoB --update $u --kl_lambda 1 --anchor_exclude $NONHO
    upd $m ${u}_kl10hoB --update $u --kl_lambda 10 --anchor_exclude $NONHO
    # focal distillation (positive-congruent training) on the training prompts
    upd $m ${u}_fd1 --update $u --fd_beta 1
    upd $m ${u}_fd10 --update $u --fd_beta 10
  done
  # milder edit (lr 3e-5, 3 epochs)
  upd $m edit_mild --update edit --lr 3e-5 --epochs 3
done

# Qwen2.5-7B only: lower learning rate, incumbent replay, LoRA rank, letter-only KL
for u in med policy; do
  upd qwen7b ${u}_lr3e-5 --update $u --lr 3e-5
  upd qwen7b ${u}_replay --update $u --replay 1000
done
upd qwen7b med_r4 --update med --rank 4
upd qwen7b med_r64 --update med --rank 64
upd qwen7b med_kl1let --update med --kl_lambda 1 --kl_mode letters
