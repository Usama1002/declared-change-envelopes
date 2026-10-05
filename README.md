# Declared Change Envelopes for Model Updates

Code for the paper "Where Did the Update Go? Certifying Declared Change Envelopes for Model Updates". An update to a language model (LoRA, full fine-tuning, DPO, GRPO, RLVR, or a knowledge editor) declares before training the slice of traffic it is meant to change. The code measures how many decisions change outside that envelope (leakage) on a pool of 34,136 multiple-choice prompts, certifies a leakage bound from unlabeled traffic with anytime-valid confidence sequences, and trains updates with envelope anchoring (KL to the incumbent at the decision position on out-of-envelope prompts). It also contains the audits on written answers, free-form generation, refusals, and open-ended chat, and the scripts that produce every table and figure of the paper.

## Repository layout

```
src/
  config.py                 data, results, checkpoint, and log directories (environment variables, repo-relative defaults)
  fp.py                     harness: prompt format, letter-level decisions and margins, embeddings, LoRA / full fine-tuning
                            with SFT, DPO, or decision-token GRPO, envelope anchoring, and the controls (L2, generic KL, ...)
  updates.py                the four scoped updates (med, legal, policy, edit) and math: training data, taxonomy envelopes,
                            natural-language scopes, in-envelope effect, option shuffling
  certify.py                Beta-mixture confidence sequence, one-sided mixture-of-SPRTs test, error spending
  build_pool.py, build_counterfact.py, build_cf_true.py, build_mc_generic.py, build_splits.py
                            traffic pool, CounterFact edits, splits, training sets, generic multiple-choice anchors
  build_generic_anchors.py  incumbent responses on generic chat / free-form prompts (token-level anchors and controls)
  run_eval.py               decisions of an incumbent or a public fine-tune on the pool (self-disagreement floor via --order_seed)
  run_update.py             train one update and evaluate it (all LoRA / full FT / DPO / GRPO variants of the paper)
  grpo_math.py              RLVR on GSM8K with free generation
  memit_edit.py             MEMIT and AlphaEdit for the edit update, with CounterFact ES/PS/NS
  make_null_adapter.py      zero and tiny random LoRA adapters (numerical floors)
  judge.py, emb_train.py    NL-judge envelope scores and training-set embeddings (declarations)
  gen_check.py              written answers vs letter-logit decisions; written-answer certificates
  freeform_check.py, freeform_judge.py   free-form slice (GSM8K, NQ-open) and its semantic judge
  chat_audit.py, chat_judge.py           open-ended chat audit (OpenAssistant prompts) and its judge
  safety_gen.py, safety_judge.py         refusals on unsafe prompts, XSTest-rubric judge
  effect_check.py           medical effect on all pool medical queries and MedQA
  analyze.py                footprints, declarations, certificates, chains, public fine-tunes -> results/summary.json
  analyze_extra.py, analyze_confirm.py, analyze_update_types.py, analyze_shift.py, kl_leak.py, sim_cs.py, sim_wor.py
                            further statistics -> results/*.json (see "Analysis" below)
scripts/                    runner scripts, numbered in execution order (one per experiment family)
paper/make_tables.py        all LaTeX tables -> paper/tables/
paper/make_figs.py          all figures -> paper/figures/
results/                    final JSON summaries used by the tables and figures (per-run files are written here as well)
data/checksums.md5          MD5 sums of the generated data files
```

## Environment

- Python 3.12, packages pinned in `requirements.txt` (the experiments ran in the NVIDIA NGC PyTorch container 25.11: torch 2.10.0a0, transformers 4.57.1, peft 0.18.0, datasets 4.4.1, accelerate 1.12.0, bitsandbytes 0.49 development build).
- NVIDIA H200 GPUs (141 GB). Every 1.5B to 32B run fits on one H200; full fine-tuning of the 7B/8B models peaks at about 95 GB. Qwen2.5-72B-Instruct needs two H200s (`export CUDA_VISIBLE_DEVICES=0,1`; the model is split layer-wise over all visible GPUs).
- Disk: about 450 GB for the Hugging Face cache (all models, including the 72B model and ten public fine-tunes), 2 GB for per-run results, about 10 GB for saved adapters.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME=/path/to/hf-cache     # set before running anything; all scripts read models and datasets from here
```

Optional directory overrides (defaults in parentheses): `DCE_DATA` (`data/`), `DCE_RESULTS` (`results/`), `DCE_CHECKPOINTS` (`checkpoints/`), `DCE_LOGS` (`logs/`).

## Models and datasets

All models and datasets are downloaded from the Hugging Face Hub on first use.

- Incumbents: `Qwen/Qwen2.5-1.5B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-32B-Instruct` (also the judge for free-form, chat, and refusal audits), `Qwen/Qwen2.5-72B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `mistralai/Mistral-7B-Instruct-v0.3`. The Llama and Mistral repositories are gated: accept their licenses on the Hub and run `huggingface-cli login` (or set `HF_TOKEN`).
- Public fine-tunes (01_base_and_floors.sh): `FreedomIntelligence/HuatuoGPT-o1-7B`, `FreedomIntelligence/HuatuoGPT-o1-8B`, `open-thoughts/OpenThinker-7B`, `chtmp223/Qwen2.5-7B-CLIPPER`, `Orion-zhen/Qwen2.5-7B-Instruct-Uncensored`, `arcee-ai/Llama-3.1-SuperNova-Lite`, `Team-ACE/ToolACE-8B`, `DeepMount00/Llama-3.1-8b-ITA`, `Vikhrmodels/Vikhr-Llama3.1-8B-Instruct-R-21-09-24`, `oumi-ai/HallOumi-8B`.
- Traffic pool and training sets: `cais/mmlu`, `allenai/ai2_arc`, `tau/commonsense_qa`, `openlifescienceai/medmcqa`, `fancyzhx/ag_news`, `google/boolq`, `stanfordnlp/sst2`, `coastalcph/lex_glue` (case_hold), `azhx/counterfact`.
- Anchors, audits, and editors: `allenai/openbookqa`, `Rowan/hellaswag`, `tatsu-lab/alpaca`, `openai/gsm8k`, `google-research-datasets/nq_open`, `GBaker/MedQA-USMLE-4-options`, `Paul/XSTest`, `mlabonne/harmful_behaviors`, `OpenAssistant/oasst2`, `Salesforce/wikitext`.

## Quick start: one update end to end

After building the data (step 1 below), this trains the medical update on Qwen2.5-1.5B-Instruct with plain LoRA and with envelope anchoring and prints the in-scope effect, the out-of-scope leakage, the anytime-valid 95% upper bound on leakage, and the smallest certified tolerance (about 10 minutes on one H200 once the model is downloaded):

```bash
bash scripts/00_data.sh
bash scripts/quickstart.sh                 # MAX_STEPS=2 bash scripts/quickstart.sh for a smoke test
```

The same steps by hand:

```bash
python3 src/run_eval.py --model qwen1.5b --family qwen1.5b --tag base --emb
python3 src/run_update.py --model qwen1.5b --update med --tag quick_med_plain
python3 src/run_update.py --model qwen1.5b --update med --kl_lambda 1 --tag quick_med_anchored
```

`run_update.py` logs a SHA-256 commitment of the declared envelope before training, trains, evaluates the decisions on the fixed evaluation half of the pool, and writes `results/<model>/<tag>.npz` (decision, margin, and option-letter probabilities per prompt; `evaluated` mask) and `results/<model>/<tag>.json` (arguments, timings, envelope hash). With `--save_adapter` the LoRA adapter goes to `checkpoints/<model>_<tag>`. Main options: `--update {med,legal,policy,edit}`, `--kl_lambda` (anchoring weight beta), `--full` (full fine-tuning), `--objective {sft,dpo,grpo}`, `--ref_beta` (GRPO reference KL), `--eval_on {subset,full,complement}`; `python3 src/run_update.py -h` lists the rest.

## Reproducing the paper

Run the scripts in order from the repository root. Each one is a plain list of commands with the exact hyperparameters of the reported runs; they run sequentially on the visible GPU(s), so independent blocks can be split across GPUs by copying lines. `SKIP_DONE=1 bash scripts/<script>.sh` skips updates whose result file already exists (useful for resuming). Approximate times are the summed job durations of the original runs on H200s, where two or three jobs usually shared one GPU.

| Step | Script | What it runs | Approx. time |
|---|---|---|---|
| 1 | `00_data.sh` | pool (34,136 prompts in 11 domains), 30/70 anchor/audit split, fixed evaluation half, training sets; checks MD5 sums | 10 to 20 min, CPU |
| 2 | `01_base_and_floors.sh` | incumbent decisions and self-disagreement floors (7B, 8B, Mistral, 1.5B), NL-judge scores, embeddings, zero/tiny adapters, ten public fine-tunes | 5 h |
| 3 | `02_lora_and_anchoring.sh` | 4 updates x 2 seeds, pilots, anchoring at beta 0.1/1/10, option-shuffled policy, edit with true facts, seed replicates, sequential chains (two orders), saved adapters for the audits | 40 h |
| 4 | `03_controls_and_baselines.sh` | lr x epoch grid, adapter scaling, L2, replay, generic chat KL, generic multiple-choice KL, KL on all traffic, judge-envelope anchoring, anchor/audit domain shift, focal distillation, rank and letter-only KL | 23 h |
| 5 | `04_scale_and_families.sh` | Qwen2.5-1.5B, Mistral-7B (with anchoring diagnostics), Qwen2.5-32B, Qwen2.5-72B (two GPUs) | 22 h |
| 6 | `05_full_finetuning.sh` | full fine-tuning, plain and anchored, three learning rates, 64/128 anchors, second seed | 11 h |
| 7 | `06_grpo_and_dpo.sh` | decision-token GRPO (reference KL 0.04 and 1, anchored, two seeds) and DPO | 7 h |
| 8 | `07_rlvr.sh` | GSM8K RLVR on 1.5B/7B/8B (plain, anchored, free-form anchored, strong), free-form audit and semantic judge | 17 h |
| 9 | `08_knowledge_editors.sh` | MEMIT and AlphaEdit with CounterFact metrics | 4 h |
| 10 | `09_confirmatory_audits.sh` | plain adapters and new anchored candidates (full FT, GRPO, 1.5B, 72B) on the never-used audit half; written-answer certificates | 6 h |
| 11 | `10_freeform_and_chat_audits.sh` | written answers, free-form slice with floors and free-form anchoring, refusals, medical effect, open-ended chat; 32B judges | 12 h |
| 12 | `11_analysis.sh` | all metrics, certificates, and summaries (see below) | about 1 h (one GPU for analyze.py) |
| 13 | `12_tables_and_figures.sh` | every table and figure | seconds, CPU |

Dependencies between scripts: 02 and later need 01; 03, 09, and 10 reuse adapters saved by 02 (`med_s0`, `policy_s0`, `chain_kl_1`, `rep_*`, `*_plainsv`, `*_anchsv`); 10 needs the zero and tiny adapters of 01; 11 needs everything before it. In total the paper uses 359 `run_update.py` runs plus 22 incumbent and public-model evaluations, 9 RLVR runs, and 6 editor runs, about 150 job-hours on H200s (roughly four to five days on two H200s).

### Analysis

`scripts/11_analysis.sh` reads the per-run `.npz` files and overwrites these summaries:

| Script | Output | Used for |
|---|---|---|
| `analyze.py` | `summary.json` | footprints, floors, declarations, certificates (20 audit orders), chains, public fine-tunes |
| `analyze_extra.py` | `extra.json` | bootstrap intervals, McNemar tests, routing vs anchoring, judge-envelope anchoring, editors, second chain order, TOST |
| `analyze_confirm.py` | `confirm.json` | confirmatory half, seed replicates, lr x epoch sweep |
| `analyze_update_types.py` | `update_types.json` | full fine-tuning, GRPO, RLVR, 1.5B and 72B, with certificates on one fixed audit order |
| `analyze_shift.py` | `shift_confirm.json` | anchor/audit domain shift, focal distillation, held-out certificates by update type |
| `kl_leak.py` | `kl_leak.json` | leakage vs out-of-scope letter KL and the pointwise flip bound for every variant |
| `sim_cs.py`, `sim_wor.py` | `sim_cs.json`, `sim_wor.json` | validity and sample cost of the certificates (simulations, CPU) |

The audit scripts of steps 8 to 11 write their own summaries directly: `gen_check_*.json`, `freeform_*.json`, `freeform_sem_*.json`, `safety3.json`, `effect_*.json`, `chat_sem_*.json`, and `results/<model>/edit_*_cfmetrics.json`.

### Tables and figures

All JSON summaries needed by the tables and figures are included in `results/`, so they can be regenerated without rerunning any experiment:

```bash
bash scripts/12_tables_and_figures.sh     # or: python3 paper/make_tables.py && python3 paper/make_figs.py
```

| Output | Source files in results/ |
|---|---|
| `footprint`, `declarations`, `public`, `scale`, `variants_{qwen,llama,other}` tables | `summary.json` |
| `anchor_main` | `summary.json`, `extra.json` |
| `controls`, `sweep`, `confirm` | `summary.json`, `confirm.json`, `extra.json` |
| `edit` | `extra.json`, `<model>/edit_*_cfmetrics.json` |
| `effect` | `effect_{qwen7b,llama8b}.json` |
| `updtypes`, `rlvr` | `update_types.json` |
| `confirm_types` | `shift_confirm.json` |
| `gencert` | `gen_check_*_confirm*.json` |
| `freeform` | `freeform_*_{v2,tiny,ffanch}.json`, `freeform_sem_*.json` |
| `safety3` | `safety3.json` |
| `chat` | `chat_sem_{qwen7b,llama8b}.json` |
| `fig_heatmap`, `fig_heatmap_public`, `fig_margin`, `fig_curves_qwen7b` | `summary.json` |
| `fig_pareto` | `summary.json`, `confirm.json` |
| `fig_kl_leak` | `kl_leak.json` |

## Troubleshooting

- Set `HF_HOME` (and `HF_TOKEN` for the gated models) before running any script. Changing it later makes `datasets` and `transformers` download everything again into the new location.
- `fp.py` disables the cuDNN scaled-dot-product-attention backend. In our software stack it returned NaN gradients on padded batches, and eager bfloat16 attention was numerically wrong for Qwen2.5 compared with an fp32 reference; the flash and math SDPA backends are used instead. Training pads on the right and evaluation on the left for the same reason.
- Decisions depend slightly on the evaluation batch size: re-evaluating the same model with a different batching changes 0.4 to 1.2% of the decisions on the pool (the self-disagreement floor reported in the paper). Keep `EVAL_BS` as set in the scripts (24; 16 for the 32B and 72B models) to compare against the shipped results; the incumbent evaluations use `run_eval.py --bs`.
- GPU nondeterminism means that rerun updates match the reported numbers up to seed-level variation, not bit for bit. On the original per-run files, `analyze_confirm.py`, `analyze_update_types.py`, `analyze_shift.py`, and `kl_leak.py` reproduce the shipped JSON files exactly. `analyze_extra.py` draws all bootstrap samples from one random stream in file order, and the shipped `extra.json` was computed before the last variants existed, so a rerun on the full set of runs moves some bootstrap interval endpoints and TOST intervals by about 0.1 points. The data files are deterministic: `00_data.sh` checks them against `data/checksums.md5`. If the CaseHOLD download fails, `build_pool.py` prints `casehold failed` and the pool (and every split) will differ.
- Out of memory: set `EVAL_BS=16` (or lower), and avoid running a full fine-tuning job next to other jobs on the same GPU. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set by `scripts/env.sh`.
- `--init` takes comma-separated adapter directories that are merged into the incumbent in order (sequential updates); `--init_nomerge` keeps a single adapter unmerged (used for the zero-adapter floor).

## License

MIT, see `LICENSE`.
