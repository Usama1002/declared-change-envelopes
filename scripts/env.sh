# Shared settings for the runner scripts (sourced, not executed).
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DATA="${DCE_DATA:-$REPO/data}"
RES="${DCE_RESULTS:-$REPO/results}"
CKPT="${DCE_CHECKPOINTS:-$REPO/checkpoints}"
PY="${PYTHON:-python3}"

# Evaluation batch size of run_update.py / grpo_math.py / memit_edit.py. Decisions depend on it only at the level of
# the self-disagreement floor; 24 was used for most runs (16 for the 32B and 72B models, set per command).
export EVAL_BS="${EVAL_BS:-24}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
if [ -z "${HF_HOME:-}" ]; then
  echo "note: HF_HOME is not set; models and datasets will be cached under ~/.cache/huggingface" >&2
fi

# run <command...>: print and run.
run() { echo "[$(date +%T)] $*"; "$@"; }

# upd <model> <tag> [run_update.py arguments...]: train and evaluate one update.
# With SKIP_DONE=1, updates whose results/<model>/<tag>.npz already exists are skipped (to resume a script).
upd() {
  local m="$1" t="$2"; shift 2
  if [ "${SKIP_DONE:-0}" = 1 ] && [ -f "$RES/$m/$t.npz" ]; then echo "skip $m/$t (done)"; return 0; fi
  run "$PY" src/run_update.py --model "$m" --tag "$t" "$@"
}
