#!/usr/bin/env bash
# RLVR on GSM8K: GRPO with LoRA, free generation (up to 320 new tokens), 16 prompts x 8 samples per step, reward 1 when
# the stated final number is correct, k3 KL to the incumbent (0.04); optional anchoring on out-of-scope multiple-choice
# traffic (--kl_lambda) and on out-of-scope free-form NQ-open prompts (--ff_anchor). Declared scope: MMLU mathematics
# and statistics. Then the free-form audit (GSM8K test, NQ-open validation) and its semantic judge (Qwen2.5-32B).
source "$(dirname "$0")/env.sh"

run "$PY" src/grpo_math.py --model qwen7b --tag math_grpo
run "$PY" src/grpo_math.py --model qwen7b --kl_lambda 1 --tag math_grpokl1
run "$PY" src/grpo_math.py --model llama8b --tag math_grpo
run "$PY" src/grpo_math.py --model llama8b --kl_lambda 1 --tag math_grpokl1
run "$PY" src/grpo_math.py --model llama8b --lr 5e-5 --steps 300 --tag math_grpostrong
run "$PY" src/grpo_math.py --model llama8b --lr 5e-5 --steps 300 --kl_lambda 1 --tag math_grpostrongkl1

# Qwen2.5-1.5B: free-form anchors from NQ-open training questions (data/anchors_nq_qwen1.5b.jsonl)
run "$PY" src/build_generic_anchors.py --model qwen1.5b --source nq --n 1500
run "$PY" src/grpo_math.py --model qwen1.5b --lr 3e-5 --steps 300 --tag math_grpo
run "$PY" src/grpo_math.py --model qwen1.5b --lr 3e-5 --steps 300 --kl_lambda 1 --tag math_grpokl1
run "$PY" src/grpo_math.py --model qwen1.5b --lr 3e-5 --steps 300 --kl_lambda 1 --ff_anchor --tag math_grpokl1ff

# free-form audit of the RLVR adapters (the 7B/8B floors are measured in 10_freeform_and_chat_audits.sh)
for m in qwen7b llama8b; do
  run "$PY" src/freeform_check.py --model $m --suffix _math --no_floor --adapters math_grpo=math_grpo,math_grpokl1=math_grpokl1
done
run "$PY" src/freeform_check.py --model llama8b --suffix _mathstrong --no_floor \
  --adapters math_grpostrong=math_grpostrong,math_grpostrongkl1=math_grpostrongkl1
run "$PY" src/freeform_check.py --model qwen1.5b --suffix _math \
  --adapters math_grpo=math_grpo,math_grpokl1=math_grpokl1,math_grpokl1ff=math_grpokl1ff
run "$PY" src/freeform_judge.py --suffix _math qwen7b llama8b qwen1.5b
run "$PY" src/freeform_judge.py --suffix _mathstrong llama8b
