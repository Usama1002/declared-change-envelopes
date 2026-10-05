"""Train one scoped update (LoRA, optional KL anchoring on declared out-of-envelope anchor traffic), then
evaluate decisions on the full traffic pool. Envelope spec is hashed and logged BEFORE training (commitment)."""
import argparse, json, os, time
import numpy as np, torch
import fp, updates as U

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen7b")
ap.add_argument("--update", required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=int, default=None)
ap.add_argument("--n_train", type=int, default=2000)
ap.add_argument("--kl_lambda", type=float, default=0.0)
ap.add_argument("--targets", default="all")
ap.add_argument("--max_steps", type=int, default=None)
ap.add_argument("--init", default="", help="comma list of adapter dirs to merge into the base before training")
ap.add_argument("--save_adapter", action="store_true")
ap.add_argument("--tag", required=True)
ap.add_argument("--permute", action="store_true", help="shuffle option order per training example")
ap.add_argument("--anchor_exclude", default="", help="comma list of domains removed from the anchor set (held-out test)")
ap.add_argument("--replay", type=int, default=0, help="add N anchor items labeled with the incumbent's own decisions")
ap.add_argument("--anchor_bs", type=int, default=None, help="anchor prompts per step (default = train batch size)")
ap.add_argument("--l2_delta", type=float, default=0.0, help="L2 penalty on the LoRA weight update (L2-to-incumbent)")
ap.add_argument("--generic_anchor", action="store_true", help="KL anchoring on generic Alpaca chat traffic instead of declared out-of-scope traffic")
ap.add_argument("--init_scale", type=float, default=1.0, help="scale applied to --init adapters before merging (weight interpolation)")
ap.add_argument("--edit_true", type=int, default=0, help="edit update: add N unedited CounterFact facts labeled with the true answer")
ap.add_argument("--init_nomerge", action="store_true", help="eval_only: keep the single --init adapter unmerged (adapter code path)")
ap.add_argument("--anchor_all", action="store_true", help="anchor on ALL anchor-split traffic, including the envelope (undeclared control)")
ap.add_argument("--anchor_file", default="", help="anchor prompts from a jsonl in data/ instead of the pool anchor split (e.g. anchors_mc_generic.jsonl)")
ap.add_argument("--ff_anchor", action="store_true", help="add token-level KL on free-form prompts (anchors_ff_{model}.jsonl) to the decision-position anchoring")
ap.add_argument("--full", action="store_true", help="full fine-tuning of all weights instead of LoRA")
ap.add_argument("--kl_mode", default="full", choices=["full", "letters"])
ap.add_argument("--objective", default="sft", choices=["sft", "dpo", "grpo"])
ap.add_argument("--ref_beta", type=float, default=0.04, help="grpo: KL(policy||reference) weight on training prompts")
ap.add_argument("--grpo_group", type=int, default=8)
ap.add_argument("--fd_beta", type=float, default=0.0, help="focal distillation (positive-congruent training) weight on training prompts")
ap.add_argument("--eval_on", default="subset", choices=["subset", "full", "complement"],
                help="which part of the pool to evaluate; complement = the half never used during development")
ap.add_argument("--envelope", default="taxonomy", choices=["taxonomy", "judge"],
                help="judge = NL-judge envelope (Qwen-7B judge scores), threshold set on the anchor split to match taxonomy mass")
ap.add_argument("--eval_only", action="store_true", help="skip training; evaluate base with --init adapters merged")
a = ap.parse_args()

out_dir = f"{fp.RES}/{a.model}"
os.makedirs(out_dir, exist_ok=True)
logf = open(f"{fp.LOGS}/{a.model}_{a.tag}.log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

pool = fp.load_pool()
split = json.load(open(f"{fp.DATA}/split.json"))


def eval_ids():
    sub = fp.eval_subset()
    if a.eval_on == "subset":
        return sub
    if a.eval_on == "full":
        return list(range(len(pool)))
    ss = set(sub)
    return [i for i in range(len(pool)) if i not in ss]
E = U.taxonomy_envelope(a.update, pool)
commit = dict(update=a.update, envelope="taxonomy", nl_scope=U.NL_SCOPE[a.update], mass_pool=float(E.mean()))
if a.envelope == "judge":
    js = np.load(f"{fp.RES}/judge_qwen7b.npz")[a.update]
    sub = np.zeros(len(pool), bool); sub[fp.eval_subset()] = True
    anc = np.zeros(len(pool), bool); anc[split["anchor"]] = True; anc &= sub  # same threshold rule as analyze_extra routing
    thr = float(np.quantile(js[anc], 1 - E[anc].mean()))
    E = js >= thr
    commit = dict(update=a.update, envelope="judge_qwen7b", nl_scope=U.NL_SCOPE[a.update], threshold=thr, mass_pool=float(E.mean()))
log("ENVELOPE COMMIT sha256=" + fp.sha(commit) + " " + json.dumps(commit))

model, tok = fp.load_model(a.model)
from peft import PeftModel
for d in [x for x in a.init.split(",") if x]:
    pm = PeftModel.from_pretrained(model, d)
    if a.init_nomerge:
        model = pm; log(f"loaded unmerged {d}"); continue
    if a.init_scale != 1.0:
        for mod in pm.modules():
            if hasattr(mod, "scaling") and isinstance(mod.scaling, dict) and "default" in mod.scaling:
                mod.scaling["default"] *= a.init_scale
    model = pm.merge_and_unload()
    log(f"merged {d}")

if a.eval_only:
    res = fp.evaluate_subset(model, tok, pool, eval_ids(), bs=int(os.environ.get("EVAL_BS", 64)))
    fp.save_eval(res, f"{out_dir}/{a.tag}.npz")
    json.dump(dict(vars(a), envelope_sha=fp.sha(commit)), open(f"{out_dir}/{a.tag}.json", "w"), indent=1)
    log("done eval_only"); raise SystemExit(0)
R, ans = U.train_set(a.update, a.n_train)
excl = set(x for x in a.anchor_exclude.split(",") if x)
anchor_ids = [i for i in split["anchor"] if (a.anchor_all or not E[i]) and pool[i]["domain"] not in excl]
anchors = [pool[i] for i in anchor_ids] if a.kl_lambda > 0 and not a.generic_anchor else None
if a.anchor_file:
    anchors = fp.load_jsonl(f"{fp.DATA}/{a.anchor_file}")
    log(f"anchors from {a.anchor_file}: {len(anchors)}")
generic = fp.load_jsonl(f"{fp.DATA}/anchors_generic_{a.model}.jsonl") if a.generic_anchor else None
if a.ff_anchor:
    generic = fp.load_jsonl(f"{fp.DATA}/anchors_ff_{a.model}.jsonl")
if a.edit_true:
    extra = fp.load_jsonl(f"{fp.DATA}/train_cf_true.jsonl")[: a.edit_true]
    R = R + extra; ans = ans + [r["answer"] for r in extra]
    log(f"edit: added {len(extra)} unedited true facts")
if a.replay:
    bdec = fp.load_eval(f"{fp.RES}/{a.model}/base.npz")["decision"]
    import random as _r
    ids = _r.Random(a.seed).sample(anchor_ids, min(a.replay, len(anchor_ids)))
    R = R + [pool[i] for i in ids]; ans = ans + [int(bdec[i]) for i in ids]
    log(f"replay: added {len(ids)} incumbent-labeled anchor items")
if a.permute:
    R, ans = U.permute_options(R, ans, seed=a.seed)
    log("permuted option order per training example")
targets = {"all": ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
           "attn": ("q_proj", "k_proj", "v_proj", "o_proj"), "mlp": ("gate_proj", "up_proj", "down_proj")}[a.targets]
rejected = None
if a.objective == "dpo":
    # rejected answer = the incumbent's highest-scoring wrong option (hard negative)
    import numpy as _np
    with torch.no_grad():
        br = fp.evaluate(model, tok, R, bs=48)
    rejected = []
    for k, (r_, y) in enumerate(zip(R, ans)):
        pr = br["probs"][k, :len(r_["choices"])].astype(float).copy(); pr[y] = -1
        rejected.append(int(_np.argmax(pr)))
    log(f"dpo: rejected = incumbent's top wrong option")
t0 = time.time()
if a.full:
    ref = None
    if anchors and a.kl_lambda > 0:
        ref, _ = fp.load_model(a.model)
        for p_ in ref.parameters():
            p_.requires_grad_(False)
    model = fp.train_full(model, tok, R, ans, seed=a.seed, lr=a.lr, epochs=a.epochs or U.default_epochs(a.update),
                          anchor_recs=anchors, kl_lambda=a.kl_lambda, max_steps=a.max_steps, log=log, anchor_bs=a.anchor_bs, ref_model=ref)
    del ref; torch.cuda.empty_cache()
    if a.save_adapter:
        a.save_adapter = False; log("full fine-tuning: weights not saved")
else:
  model = fp.train_lora(model, tok, R, ans, seed=a.seed, rank=a.rank, alpha=2 * a.rank, lr=a.lr,
                        epochs=a.epochs or U.default_epochs(a.update), anchor_recs=anchors, kl_lambda=a.kl_lambda,
                        targets=targets, max_steps=a.max_steps, log=log, objective=a.objective, rejected=rejected, kl_mode=a.kl_mode, anchor_bs=a.anchor_bs,
                        l2_delta=a.l2_delta, generic_anchors=generic, grpo_group=a.grpo_group, ref_beta=a.ref_beta, fd_beta=a.fd_beta)
train_time = time.time() - t0
if a.save_adapter:
    ad = f"{fp.CKPT}/{a.model}_{a.tag}"
    model.save_pretrained(ad); log(f"saved adapter {ad}")
t0 = time.time()
res = fp.evaluate_subset(model, tok, pool, eval_ids(), bs=int(os.environ.get("EVAL_BS", 64)))
fp.save_eval(res, f"{out_dir}/{a.tag}.npz")
meta = dict(vars(a), train_time=train_time, eval_time=time.time() - t0, envelope_sha=fp.sha(commit))
json.dump(meta, open(f"{out_dir}/{a.tag}.json", "w"), indent=1)
log("done " + json.dumps(meta))
