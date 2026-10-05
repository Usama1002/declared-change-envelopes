#!/usr/bin/env bash
# Audits beyond the letter-level decision, all with greedy decoding: written answers to the multiple-choice prompts,
# free-form traffic (GSM8K final numbers, NQ-open short answers), free-form anchoring, refusals on unsafe prompts,
# medical effect with adequate power (all pool medical queries + MedQA), and open-ended chat (2,000 OpenAssistant
# prompts). Judges: Qwen2.5-32B-Instruct. Requires 01 and 02.
source "$(dirname "$0")/env.sh"

# written answers vs letter-logit decisions on 2,000 development audit queries
run "$PY" src/gen_check.py --model qwen7b --adapters med_s0,policy_s0,chain_kl_1
run "$PY" src/gen_check.py --model llama8b --adapters med_s0,policy_s0
run "$PY" src/gen_check.py --model qwen7b --scope_update med --adapters med_s0,chain_kl_1
run "$PY" src/gen_check.py --model qwen7b --scope_update policy --adapters policy_s0,rep_policy_kl1_seed1
run "$PY" src/gen_check.py --model llama8b --scope_update med --adapters med_s0,rep_med_kl1_seed1
run "$PY" src/gen_check.py --model llama8b --scope_update policy --adapters policy_s0,rep_policy_kl1_seed1
for m in qwen7b llama8b; do
  for u in med policy; do run "$PY" src/gen_check.py --model $m --scope_update $u --adapters "" --floor; done
done

# free-form slice: floors (re-batched incumbent, zero adapter, tiny random adapters) and plain/anchored updates
run "$PY" src/freeform_check.py --model qwen7b --suffix _v2 \
  --adapters null=null_adapter,med_plain=med_s0,med_anch=chain_kl_1,policy_plain=policy_s0,policy_anch=rep_policy_kl1_seed1
run "$PY" src/freeform_check.py --model llama8b --suffix _v2 \
  --adapters null=null_adapter,med_plain=med_s0,med_anch=rep_med_kl1_seed1,policy_plain=policy_s0,policy_anch=rep_policy_kl1_seed1
for m in qwen7b llama8b; do
  run "$PY" src/freeform_check.py --model $m --suffix _tiny --no_floor --adapters tiny4=tiny_adapter_0.0001,tiny3=tiny_adapter_0.001
done

# free-form anchoring: decision-position KL plus token-level KL on 1,500 out-of-scope free-form prompts
# (GSM8K train and NQ-open train, data/anchors_ff_<model>.jsonl)
for m in qwen7b llama8b; do
  run "$PY" src/build_generic_anchors.py --model $m --source freeform --n 1500
  upd $m med_mcffkl1 --update med --kl_lambda 1 --ff_anchor --save_adapter
  upd $m policy_mcffkl1 --update policy --kl_lambda 1 --ff_anchor --save_adapter
  run "$PY" src/freeform_check.py --model $m --suffix _ffanch --no_floor --adapters med_mcff=med_mcffkl1,policy_mcff=policy_mcffkl1
done
run "$PY" src/freeform_judge.py qwen7b llama8b
run "$PY" src/freeform_judge.py --suffix _ffanch qwen7b llama8b

# refusals on 720 unsafe prompts (XSTest unsafe + AdvBench), LLM-judged with the XSTest rubric
run "$PY" src/safety_gen.py --model qwen7b \
  --adapters plain_s0=med_s0,plain_s1=med_s1,anch_s0=chain_kl_1,anch_s1=rep_med_kl1_seed1,anch_s2=rep_med_kl1_seed2
run "$PY" src/safety_gen.py --model llama8b \
  --adapters plain_s0=med_s0,plain_s1=med_s1,anch_s1=rep_med_kl1_seed1,anch_s2=rep_med_kl1_seed2,anch_s3=chain_kl_1
run "$PY" src/safety_judge.py qwen7b llama8b

# medical effect with adequate power
run "$PY" src/effect_check.py --model qwen7b \
  --adapters plain_s0=med_s0,plain_s1=med_s1,kl1_s0=chain_kl_1,kl1_s1=rep_med_kl1_seed1,kl1_s2=rep_med_kl1_seed2,kl10_s1=rep_med_kl10_seed1,kl10_s2=rep_med_kl10_seed2
run "$PY" src/effect_check.py --model llama8b \
  --adapters plain_s0=med_s0,plain_s1=med_s1,kl1_s1=rep_med_kl1_seed1,kl1_s2=rep_med_kl1_seed2,kl10_s1=rep_med_kl10_seed1,kl10_s2=rep_med_kl10_seed2

# open-ended chat: responses (256 new tokens), then scope and substance judged by Qwen2.5-32B
for m in qwen7b llama8b; do run "$PY" src/chat_audit.py --model $m; done
run "$PY" src/chat_judge.py qwen7b llama8b
