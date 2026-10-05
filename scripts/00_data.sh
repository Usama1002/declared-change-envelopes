#!/usr/bin/env bash
# Build the traffic pool, the anchor/audit split, the fixed evaluation half, and the update training sets (CPU only,
# downloads public datasets from the Hugging Face Hub). Outputs go to $DCE_DATA (default: data/).
source "$(dirname "$0")/env.sh"

run "$PY" src/build_pool.py          # pool.jsonl (MMLU, ARC, CSQA, MedMCQA, AG News, BoolQ, SST-2, CaseHOLD) + train_{med,legal,agnews}.jsonl
run "$PY" src/build_counterfact.py   # train_cf.jsonl (400 edits) + pool_cf.jsonl (paraphrase and neighborhood prompts)
run "$PY" src/build_cf_true.py       # train_cf_true.jsonl (400 unedited facts, used by --edit_true)
run "$PY" src/build_mc_generic.py    # anchors_mc_generic.jsonl (OpenBookQA + HellaSwag, used by --anchor_file)
run "$PY" src/build_splits.py        # merges pool_cf into pool.jsonl; split.json (30% anchor / 70% audit), subset.json

# The files are byte-identical to the ones used in the paper.
(cd "$DATA" && md5sum -c "$REPO/data/checksums.md5")
