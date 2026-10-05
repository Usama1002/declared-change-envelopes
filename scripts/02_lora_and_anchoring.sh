#!/usr/bin/env bash
# LoRA updates (rank 16, alpha 32, lr 1e-4, batch 16, 2 epochs; 5 epochs for edit; 2,000 training examples) and
# envelope anchoring (--kl_lambda = beta: KL to the incumbent at the decision position on anchor-split prompts outside
# the declared envelope). Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct. Requires 01_base_and_floors.sh.
source "$(dirname "$0")/env.sh"

for m in qwen7b llama8b; do
  # plain LoRA, two seeds per update (adapters are reused by later scripts)
  for u in med legal policy edit; do
    upd $m ${u}_s0 --update $u --save_adapter
    upd $m ${u}_s1 --update $u --seed 1 --save_adapter
  done
  # short pilot runs (seed 2) used by the pilot-forecast declaration
  for u in med legal policy; do upd $m pilot_$u --update $u --seed 2 --max_steps 62; done
  upd $m pilot_edit --update edit --seed 2 --max_steps 32
  # anchored
  for u in med legal policy edit; do upd $m ${u}_kl1 --update $u --kl_lambda 1; done
  for u in med policy; do upd $m ${u}_kl10 --update $u --kl_lambda 10; done
  # option-shuffled policy (letters carry no information about content), plain and anchored
  upd $m policy_perm --update policy --permute
  upd $m policy_permkl1 --update policy --kl_lambda 1 --permute
  # edit with 400 unedited true facts mixed in, plain and anchored
  upd $m edit_mix --update edit --edit_true 400
  upd $m edit_mixkl1 --update edit --kl_lambda 1 --edit_true 400
done
for u in med policy; do upd qwen7b ${u}_kl0.1 --update $u --kl_lambda 0.1; done
upd qwen7b edit_perm --update edit --permute

# seed replicates of anchored updates, evaluated on the whole pool (development half and confirmatory half)
for s in 1 2; do
  for m in qwen7b llama8b; do
    upd $m rep_med_kl1_seed$s --update med --seed $s --kl_lambda 1 --eval_on full --save_adapter
    upd $m rep_med_kl10_seed$s --update med --seed $s --kl_lambda 10 --eval_on full --save_adapter
    upd $m rep_policy_kl1_seed$s --update policy --seed $s --kl_lambda 1 --eval_on full --save_adapter
    upd $m rep_legal_kl1_seed$s --update legal --seed $s --kl_lambda 1 --eval_on full --save_adapter
  done
  upd qwen7b rep_policy_kl10_seed$s --update policy --seed $s --kl_lambda 10 --eval_on full --save_adapter
  upd qwen7b rep_edit_kl1_seed$s --update edit --seed $s --kl_lambda 1 --eval_on full --save_adapter
done

# sequential composition: med -> legal -> policy -> edit, plain and anchored (beta 1); the last step also with true facts
for m in qwen7b llama8b; do
  C=$CKPT/$m
  upd $m chain_plain_2 --update legal --init ${C}_med_s0 --save_adapter
  upd $m chain_plain_3 --update policy --init ${C}_med_s0,${C}_chain_plain_2 --save_adapter
  upd $m chain_plain_4 --update edit --init ${C}_med_s0,${C}_chain_plain_2,${C}_chain_plain_3
  upd $m chain_plain_4mix --update edit --edit_true 400 --init ${C}_med_s0,${C}_chain_plain_2,${C}_chain_plain_3
  upd $m chain_kl_1 --update med --kl_lambda 1 --save_adapter
  upd $m chain_kl_2 --update legal --kl_lambda 1 --init ${C}_chain_kl_1 --save_adapter
  upd $m chain_kl_3 --update policy --kl_lambda 1 --init ${C}_chain_kl_1,${C}_chain_kl_2 --save_adapter
  upd $m chain_kl_4 --update edit --kl_lambda 1 --init ${C}_chain_kl_1,${C}_chain_kl_2,${C}_chain_kl_3
  upd $m chain_kl_4mix --update edit --kl_lambda 1 --edit_true 400 --init ${C}_chain_kl_1,${C}_chain_kl_2,${C}_chain_kl_3
done
# second order on Qwen: policy -> edit (with true facts) -> legal -> med
C=$CKPT/qwen7b
upd qwen7b c2_plain_2 --update edit --edit_true 400 --init ${C}_policy_s0 --save_adapter
upd qwen7b c2_plain_3 --update legal --init ${C}_policy_s0,${C}_c2_plain_2 --save_adapter
upd qwen7b c2_plain_4 --update med --init ${C}_policy_s0,${C}_c2_plain_2,${C}_c2_plain_3
upd qwen7b c2_kl_1 --update policy --kl_lambda 1 --save_adapter
upd qwen7b c2_kl_2 --update edit --kl_lambda 1 --edit_true 400 --init ${C}_c2_kl_1 --save_adapter
upd qwen7b c2_kl_3 --update legal --kl_lambda 1 --init ${C}_c2_kl_1,${C}_c2_kl_2 --save_adapter
upd qwen7b c2_kl_4 --update med --kl_lambda 1 --init ${C}_c2_kl_1,${C}_c2_kl_2,${C}_c2_kl_3

# saved plain/anchored adapters for the written-answer and chat audits (anchoring beta 10 for med and policy, 1 for legal)
for m in qwen7b llama8b; do
  upd $m med_plainsv --update med --save_adapter
  upd $m med_anchsv --update med --kl_lambda 10 --save_adapter
  upd $m legal_plainsv --update legal --save_adapter
  upd $m legal_anchsv --update legal --kl_lambda 1 --save_adapter
  upd $m policy_plainsv --update policy --save_adapter
  upd $m policy_anchsv --update policy --kl_lambda 10 --save_adapter
done
