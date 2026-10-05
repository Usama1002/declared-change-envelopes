#!/usr/bin/env bash
# Incumbent decisions on the full pool (with embeddings), self-disagreement floors (same model, different batching),
# NL-judge envelope scores, training-set embeddings, zero and tiny random adapters, and the public fine-tunes.
# The 32B and 72B incumbents are evaluated in 04_scale_and_families.sh.
source "$(dirname "$0")/env.sh"

# incumbents and the re-batched floor (null_order1)
for m in qwen7b llama8b; do
  run "$PY" src/run_eval.py --model $m --family $m --tag base --emb
  run "$PY" src/run_eval.py --model $m --family $m --tag null_order1 --bs 32 --order_seed 1
done
run "$PY" src/run_eval.py --model mistral7b --family mistral7b --tag base --emb --bs 24
run "$PY" src/run_eval.py --model mistral7b --family mistral7b --tag null_order1 --bs 16 --order_seed 1
run "$PY" src/run_eval.py --model qwen1.5b --family qwen1.5b --tag base --emb
run "$PY" src/run_eval.py --model qwen1.5b --family qwen1.5b --tag null_order1 --order_seed 1

# NL-judge envelope scores (Qwen2.5-7B-Instruct as judge) and update training-set embeddings (data kNN declarations)
run "$PY" src/judge.py --judge qwen7b
for m in qwen7b llama8b; do run "$PY" src/emb_train.py --model $m; done

# zero adapter (adapter code path floor) and tiny random adapters (sigma_B = 1e-4, 1e-3)
for m in qwen7b llama8b; do
  run "$PY" src/make_null_adapter.py $m
  run "$PY" src/make_null_adapter.py $m 0.0001
  run "$PY" src/make_null_adapter.py $m 0.001
  upd $m null_adapter --update med --init $CKPT/${m}_null_adapter --init_nomerge --eval_only
done

# public fine-tunes of the same incumbents (evaluated on the fixed evaluation half)
for id in FreedomIntelligence/HuatuoGPT-o1-7B open-thoughts/OpenThinker-7B chtmp223/Qwen2.5-7B-CLIPPER; do
  run "$PY" src/run_eval.py --model $id --family qwen7b --tag pub_${id#*/}
done
run "$PY" src/run_eval.py --model Orion-zhen/Qwen2.5-7B-Instruct-Uncensored --family qwen7b --tag pub_Qwen2.5-7B-Instruct-Uncensored --bs 24
for id in arcee-ai/Llama-3.1-SuperNova-Lite Team-ACE/ToolACE-8B Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24 \
          oumi-ai/HallOumi-8B FreedomIntelligence/HuatuoGPT-o1-8B; do
  run "$PY" src/run_eval.py --model $id --family llama8b --tag pub_${id#*/}
done
run "$PY" src/run_eval.py --model DeepMount00/Llama-3.1-8b-ITA --family llama8b --tag pub_Llama-3.1-8b-ITA --bs 24
