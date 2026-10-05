#!/usr/bin/env bash
# Locality-preserving knowledge editors for the edit update: MEMIT and AlphaEdit (down_proj of layers 4-8, the same
# 400 CounterFact rewrites, covariance from 1M wikitext-103 tokens, cached in $DCE_CHECKPOINTS), with the standard
# CounterFact ES/PS/NS metrics (--cf_metrics, written to results/<model>/<tag>_cfmetrics.json).
source "$(dirname "$0")/env.sh"

for m in qwen7b llama8b; do
  run "$PY" src/memit_edit.py --model $m --method memit --cf_metrics
  run "$PY" src/memit_edit.py --model $m --method alphaedit --cf_metrics
done
# covariance-weight variants (default weight 15,000)
run "$PY" src/memit_edit.py --model qwen7b --method memit --mom2_weight 2000 --tag edit_memitw2k
run "$PY" src/memit_edit.py --model llama8b --method memit --mom2_weight 60000 --tag edit_memitw60k --cf_metrics
