#!/usr/bin/env bash
# Preference and RL objectives on the decision token. GRPO: 8 sampled answer letters per prompt, reward 1 for the
# target letter, group-normalized advantages, KL(policy || incumbent) on the training prompts with weight --ref_beta
# (default 0.04); optionally anchored. DPO: rejected answer = the incumbent's top wrong option, beta 0.1.
source "$(dirname "$0")/env.sh"

for m in qwen7b llama8b; do
  for u in med policy; do
    upd $m ${u}_grpo --update $u --objective grpo
    upd $m ${u}_grporef1 --update $u --objective grpo --ref_beta 1
    upd $m ${u}_grpokl1 --update $u --objective grpo --kl_lambda 1
    upd $m ${u}_grpos1 --update $u --objective grpo --seed 1
    upd $m ${u}_grporef1s1 --update $u --objective grpo --ref_beta 1 --seed 1
    upd $m ${u}_grpokl1s1 --update $u --objective grpo --kl_lambda 1 --seed 1
  done
done

upd qwen7b med_dpo --update med --objective dpo --save_adapter
upd qwen7b med_dpokl1 --update med --objective dpo --kl_lambda 1
upd qwen7b policy_dpo --update policy --objective dpo --save_adapter
upd qwen7b policy_dpokl1 --update policy --objective dpo --kl_lambda 1
upd llama8b med_dpo --update med --objective dpo --save_adapter
upd llama8b med_dpokl1 --update med --objective dpo --kl_lambda 1
