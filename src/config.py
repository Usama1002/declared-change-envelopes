"""Directory layout. Every path can be overridden with an environment variable; defaults are inside the repository.

  DCE_DATA         traffic pool, splits, training sets, anchor files    (default: <repo>/data)
  DCE_RESULTS      per-run decision files (.npz), metadata, summaries   (default: <repo>/results)
  DCE_CHECKPOINTS  saved LoRA adapters and editor statistics            (default: <repo>/checkpoints)
  DCE_LOGS         training logs                                        (default: <repo>/logs)

The Hugging Face cache is controlled by the standard HF_HOME variable; set it before running any script.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("DCE_DATA", os.path.join(REPO, "data"))
RESULTS = os.environ.get("DCE_RESULTS", os.path.join(REPO, "results"))
CKPT = os.environ.get("DCE_CHECKPOINTS", os.path.join(REPO, "checkpoints"))
LOGS = os.environ.get("DCE_LOGS", os.path.join(REPO, "logs"))
for _d in (DATA, RESULTS, CKPT, LOGS):
    os.makedirs(_d, exist_ok=True)
