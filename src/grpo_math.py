"""RLVR update: GRPO with LoRA on GSM8K training problems, free generation, verifiable reward (the stated
final number equals the gold answer). Standard GRPO: G samples per prompt, group-normalized advantages, on-policy
single update per batch (importance ratio 1), per-token k3 KL to the incumbent (adapter disabled) with weight ref_beta.
Optional anchoring: decision-position KL to the incumbent on out-of-scope multiple-choice anchor traffic (declared
scope: MMLU mathematics/statistics). Afterwards the decisions on the traffic pool are evaluated like every update."""
import argparse, json, math, os, random, re, time
import numpy as np, torch, torch.nn.functional as F
import fp, updates as U
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen7b")
ap.add_argument("--tag", required=True)
ap.add_argument("--steps", type=int, default=150)
ap.add_argument("--prompts", type=int, default=16)
ap.add_argument("--group", type=int, default=8)
ap.add_argument("--max_new", type=int, default=320)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--rank", type=int, default=16)
ap.add_argument("--ref_beta", type=float, default=0.04)
ap.add_argument("--kl_lambda", type=float, default=0.0, help="anchoring weight on out-of-scope MC anchor traffic")
ap.add_argument("--micro", type=int, default=16)
ap.add_argument("--ff_anchor", action="store_true", help="also token-level KL to the incumbent on out-of-scope free-form prompts (anchors_nq_{model}.jsonl)")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

logf = open(f"{fp.LOGS}/{a.model}_{a.tag}.log", "w")
def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()

pool = fp.load_pool(); split = json.load(open(f"{fp.DATA}/split.json"))
E = U.taxonomy_envelope("math", pool)
commit = dict(update="math", envelope="taxonomy", nl_scope=U.NL_SCOPE["math"], sources=sorted(U.MATH_SOURCES), mass_pool=float(E.mean()))
log("ENVELOPE COMMIT sha256=" + fp.sha(commit) + " " + json.dumps(commit))
anchors = [pool[i] for i in split["anchor"] if not E[i]]
ffa = fp.load_jsonl(f"{fp.DATA}/anchors_nq_{a.model}.jsonl") if a.ff_anchor else None

TEMPLATE = "{q}\n\nSolve the problem step by step, then finish with 'The answer is N.' where N is a number."
train = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=a.seed)
def gold(r):
    return r["answer"].split("####")[-1].strip().replace(",", "")
def parse(s):
    m = re.findall(r"answer is\W*?(?:N\W*)?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", s.replace("**", ""))
    if not m:
        return None
    x = m[-1].replace(",", "")
    try:
        v = float(x); return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return x
def gnorm(x):
    v = float(x); return str(int(v)) if v == int(v) else str(v)

from peft import LoraConfig, get_peft_model
fp.seed_all(a.seed)
model, tok = fp.load_model(a.model)
cfg = LoraConfig(r=a.rank, lora_alpha=2 * a.rank, lora_dropout=0.0, task_type="CAUSAL_LM",
                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, cfg)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 10))
rng = random.Random(a.seed)


def seq_logps(m, ids, att, resp_mask):
    """Per-token log-probs of ids[:,1:] under m, only where resp_mask[:,1:] (response tokens)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = m(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    lp = torch.log_softmax(logits, -1).gather(-1, ids[:, 1:, None])[..., 0]
    return lp * resp_mask[:, 1:]


t0 = time.time(); ptr = 0; hist = []
for step in range(a.steps):
    rows = [train[(ptr + k) % len(train)] for k in range(a.prompts)]; ptr += a.prompts
    texts = [tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(q=r["question"])}], tokenize=False, add_generation_prompt=True) for r in rows]
    tok.padding_side = "left"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
    model.eval(); model.config.use_cache = True
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        g = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=a.max_new,
                           num_return_sequences=a.group, pad_token_id=tok.pad_token_id)
    model.train(); model.config.use_cache = False
    P = enc["input_ids"].shape[1]
    resp = g[:, P:]
    dec = tok.batch_decode(resp, skip_special_tokens=True)
    r = torch.tensor([1.0 if (parse(t) is not None and parse(t) == gnorm(gold(rows[k // a.group]))) else 0.0 for k, t in enumerate(dec)], device="cuda")
    rg = r.view(a.prompts, a.group)
    adv = ((rg - rg.mean(1, keepdim=True)) / (rg.std(1, keepdim=True) + 1e-4)).view(-1)
    # response mask: tokens up to and including the first eos/pad
    eos_ids = {tok.eos_token_id, tok.pad_token_id} | set(getattr(model.generation_config, "eos_token_id", []) if isinstance(getattr(model.generation_config, "eos_token_id", None), list) else [])
    rc = resp.cpu().numpy()
    is_eos = np.isin(rc, list(eos_ids))
    first = np.where(is_eos.any(1), is_eos.argmax(1), rc.shape[1])
    rm = torch.tensor((np.arange(rc.shape[1])[None, :] <= first[:, None]).astype(np.float32), device="cuda")
    att = torch.cat([enc["attention_mask"].repeat_interleave(a.group, 0), rm.long()], 1)
    full_mask = torch.cat([torch.zeros(g.shape[0], P, device="cuda"), rm], 1)
    ntok = rm.sum(1).clamp(min=1)
    opt.zero_grad(set_to_none=True)
    pg_tot = kl_tot = 0.0
    for b in range(0, g.shape[0], a.micro):
        sl = slice(b, b + a.micro)
        with torch.no_grad():
            with model.disable_adapter():
                ref = seq_logps(model, g[sl], att[sl], full_mask[sl])
        cur = seq_logps(model, g[sl], att[sl], full_mask[sl])
        m_ = full_mask[sl][:, 1:]
        d = (ref - cur) * m_
        k3 = (torch.exp(d) - d - 1) * m_                                  # per-token k3 estimator of KL(pi || ref)
        per_seq = (-(adv[sl, None] * cur) + a.ref_beta * k3).sum(1) / ntok[sl]
        loss = per_seq.sum() / g.shape[0]
        loss.backward()
        pg_tot += float((-(adv[sl, None] * cur)).sum(1).div(ntok[sl]).sum()) / g.shape[0]
        kl_tot += float(k3.sum(1).div(ntok[sl]).sum()) / g.shape[0]
    akl = 0.0
    if a.kl_lambda > 0:
        aa = rng.sample(range(len(anchors)), a.prompts)
        aenc, _ = fp.build_batch(tok, [anchors[i] for i in aa], [0] * len(aa)); aenc = aenc.to("cuda")
        with torch.no_grad():
            with model.disable_adapter():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    bl = torch.log_softmax(fp.last_logits(model, aenc), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl = torch.log_softmax(fp.last_logits(model, aenc), -1)
        kl = (bl.exp() * (bl - cl)).sum(-1).mean()
        (a.kl_lambda * kl).backward(); akl = float(kl)
    if ffa is not None:
        aa = rng.sample(range(len(ffa)), a.prompts)
        seqs = [ffa[i]["prompt_ids"][-448:] + ffa[i]["resp_ids"] for i in aa]
        L = max(len(x) for x in seqs)
        ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(seqs), L), dtype=torch.long)
        msk = torch.zeros((len(seqs), L), dtype=torch.bool)
        for j, (x, i) in enumerate(zip(seqs, aa)):
            ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
            p0 = len(x) - len(ffa[i]["resp_ids"]); msk[j, p0 - 1:len(x) - 1] = True
        ids, att, msk = ids.cuda(), att.cuda(), msk.cuda()
        with torch.no_grad():
            with model.disable_adapter():
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    bl = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[msk].float(), -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            cl = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[msk].float(), -1)
        fkl = (bl.exp() * (bl - cl)).sum(-1).mean()
        (a.kl_lambda * fkl).backward(); akl += float(fkl)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    opt.step(); sched.step()
    hist.append(dict(step=step, reward=float(r.mean()), ref_kl=kl_tot, anchor_kl=akl, len=float(ntok.mean())))
    if step % 10 == 0 or step == a.steps - 1:
        log(f"step {step}/{a.steps} reward {r.mean():.3f} ref_kl {kl_tot:.4f} anchor_kl {akl:.4f} len {ntok.mean():.0f} t {time.time()-t0:.0f}s")

train_time = time.time() - t0
model.eval(); model.config.use_cache = True
ad = f"{fp.CKPT}/{a.model}_{a.tag}"
model.save_pretrained(ad); log(f"saved adapter {ad}")
t0 = time.time()
res = fp.evaluate_subset(model, tok, pool, fp.eval_subset(), bs=int(os.environ.get("EVAL_BS", 24)))
os.makedirs(f"{fp.RES}/{a.model}", exist_ok=True)
fp.save_eval(res, f"{fp.RES}/{a.model}/{a.tag}.npz")
meta = dict(vars(a), update="math", train_time=train_time, eval_time=time.time() - t0, envelope_sha=fp.sha(commit), history=hist)
json.dump(meta, open(f"{fp.RES}/{a.model}/{a.tag}.json", "w"), indent=1)
log("done")
