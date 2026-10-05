"""Locality-preserving knowledge-editing baselines for the edit update: MEMIT (Meng et al., 2023) and
AlphaEdit (Fang et al., 2025), re-implemented for Qwen2.5 / Llama-3.1 (MLP down_proj of layers 4-8). The same 400
CounterFact rewrites used by the LoRA edit update are inserted in one batch; the edited model is then evaluated with
the standard decision harness and saved as results/{model}/edit_{method}.npz, so the footprint analysis treats it like
any other edit variant.

Algorithm (MEMIT): (1) second-moment statistics C_l = E[k k^T] of down_proj inputs over wikitext-103 tokens; (2) for
each rewrite, a target residual z at the last edited layer L, found by optimizing a vector added to the layer-L output
at the subject's last token (NLL of the new object plus a KL term on "{subject} is a"); (3) layer-by-layer least-squares
spreading: Delta_l = R_l K_l^T (lambda C_l + K_l K_l^T)^{-1}. AlphaEdit replaces (3) by the null-space-projected
update Delta_l = R_l K_l^T P_l (K_l K_l^T P_l + l2 I)^{-1}, with P_l projecting onto the eigenvectors of C_l whose
eigenvalues are below 1e-2."""
import argparse, json, os, time
import numpy as np, torch
import torch.nn.functional as F
import fp
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="qwen7b")
ap.add_argument("--method", default="memit", choices=["memit", "alphaedit"])
ap.add_argument("--layers", default="4,5,6,7,8")
ap.add_argument("--n_edits", type=int, default=400)
ap.add_argument("--cov_tokens", type=int, default=1_000_000)
ap.add_argument("--mom2_weight", type=float, default=15000.0)
ap.add_argument("--l2", type=float, default=10.0)
ap.add_argument("--null_thr", type=float, default=1e-2)
ap.add_argument("--v_steps", type=int, default=25)
ap.add_argument("--v_lr", type=float, default=0.1)
ap.add_argument("--clamp", type=float, default=4.0)
ap.add_argument("--kl_factor", type=float, default=0.0625)
ap.add_argument("--wd", type=float, default=1e-3)
ap.add_argument("--tag", default=None)
ap.add_argument("--no_eval", action="store_true")
ap.add_argument("--cf_metrics", action="store_true", help="standard CounterFact ES/PS/NS (raw-text completion) before and after editing")
ap.add_argument("--eval_on", default="subset", choices=["subset", "full"])
a = ap.parse_args()
tag = a.tag or f"edit_{a.method}"
LAYERS = [int(x) for x in a.layers.split(",")]
L_LAST = LAYERS[-1]
os.makedirs(f"{fp.RES}/{a.model}", exist_ok=True)
logf = open(f"{fp.LOGS}/{a.model}_{tag}.log", "w")


def log(s):
    print(s, flush=True); logf.write(s + "\n"); logf.flush()


model, tok = fp.load_model(a.model)
for p in model.parameters():
    p.requires_grad_(False)
layers = model.model.layers
down = {l: layers[l].mlp.down_proj for l in LAYERS}
dev = "cuda"

# ------------------------------------------------------------------ rewrites
recs = fp.load_jsonl(f"{fp.DATA}/train_cf.jsonl")[: a.n_edits]
reqs = []
for r in recs:
    prompt = r["question"].replace("Complete the statement: ", "")
    assert r["subject"] in prompt, r
    reqs.append(dict(prompt=prompt, subject=r["subject"], target=" " + r["choices"][r["answer"]]))
CTX = ["{}", "The following is a fact. {}", "According to the encyclopedia, {}", "Here is what I know. {}",
       "In summary, {}", "Background: {}"]


def subj_last_index(text, subject):
    """Token index of the last subject token in text (no special tokens)."""
    start = text.index(subject)
    pre = tok(text[: start + len(subject)], add_special_tokens=False)["input_ids"]
    return len(pre) - 1


# ------------------------------------------------------------------ second-moment statistics
def cov_stats():
    cache = f"{fp.CKPT}/{a.model}_mom2_{'-'.join(map(str, LAYERS))}_{a.cov_tokens}.pt"
    if os.path.exists(cache):
        return torch.load(cache)
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train[:300000]")
    text = "\n".join(t for t in ds["text"] if len(t) > 50)
    ids = tok(text[: a.cov_tokens * 6], add_special_tokens=False)["input_ids"][: a.cov_tokens]
    d_in = down[LAYERS[0]].in_features
    C = {l: torch.zeros(d_in, d_in, device=dev, dtype=torch.float32) for l in LAYERS}
    cnt = 0
    hooks = []
    for l in LAYERS:
        def pre(mod, inp, l=l):
            x = inp[0].reshape(-1, inp[0].shape[-1]).float()
            C[l].addmm_(x.T, x)
        hooks.append(down[l].register_forward_pre_hook(pre))
    seq = 512
    chunks = [ids[i: i + seq] for i in range(0, len(ids) - seq + 1, seq)]
    with torch.no_grad():
        for b in range(0, len(chunks), 8):
            x = torch.tensor(chunks[b: b + 8], device=dev)
            model.model(input_ids=x)  # hidden states only
            cnt += x.numel()
    for h in hooks:
        h.remove()
    C = {l: (C[l] / cnt).cpu() for l in LAYERS}
    torch.save(C, cache)
    log(f"covariance from {cnt} tokens")
    return C


# ------------------------------------------------------------------ keys and targets
def batch_enc(texts):
    tok.padding_side = "right"
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    tok.padding_side = "left"
    return enc


@torch.no_grad()
def compute_keys(l, reqs):
    """Mean down_proj input at the subject's last token over context templates: (d_in, n)."""
    K = []
    for r in reqs:
        texts = [c.format(r["prompt"]) for c in CTX]
        idx = [subj_last_index(t, r["subject"]) for t in texts]
        cap = {}
        h = down[l].register_forward_pre_hook(lambda m, i: cap.__setitem__("k", i[0]))
        model.model(**batch_enc(texts))
        h.remove()
        K.append(torch.stack([cap["k"][j, idx[j]] for j in range(len(texts))]).float().mean(0))
    return torch.stack(K, 1)


@torch.no_grad()
def layer_out(l, reqs):
    """Residual stream after layer l at the subject's last token (plain prompt): (hidden, n)."""
    Z = []
    for b in range(0, len(reqs), 32):
        rr = reqs[b: b + 32]
        texts = [r["prompt"] for r in rr]
        idx = [subj_last_index(t, r["subject"]) for t, r in zip(texts, rr)]
        cap = {}
        h = layers[l].register_forward_hook(lambda m, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
        model.model(**batch_enc(texts))
        h.remove()
        Z.append(torch.stack([cap["h"][j, idx[j]] for j in range(len(rr))]).float())
    return torch.cat(Z, 0).T


def compute_z(r):
    """Optimize delta added to layer-L output at the subject's last token so that the new object becomes likely."""
    texts = [c.format(r["prompt"]) + r["target"] for c in CTX]
    kl_text = f"{r['subject']} is a"
    enc = batch_enc(texts + [kl_text])
    tgt_ids = tok(r["target"], add_special_tokens=False)["input_ids"]
    lens = enc["attention_mask"].sum(1)
    sidx = [subj_last_index(t, r["subject"]) for t in texts] + [subj_last_index(kl_text, r["subject"])]
    hid = model.config.hidden_size
    delta = torch.zeros(hid, device=dev, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=a.v_lr)
    state = {}

    def hook(m, i, o):
        h = o[0] if isinstance(o, tuple) else o
        if "h0" not in state:
            state["h0"] = h[0, sidx[0]].detach().float().clone()
        h = h.clone()
        for j, s in enumerate(sidx):
            h[j, s] = h[j, s] + delta.to(h.dtype)
        return (h,) + tuple(o[1:]) if isinstance(o, tuple) else h

    kl0 = None
    for step in range(a.v_steps):
        hk = layers[L_LAST].register_forward_hook(hook)
        logits = model(**enc).logits.float()
        hk.remove()
        lp = torch.log_softmax(logits, -1)
        nll = 0.0
        for j in range(len(texts)):
            end = int(lens[j]); T = len(tgt_ids)
            pos = torch.arange(end - T - 1, end - 1, device=dev)
            nll = nll - lp[j, pos, torch.tensor(tgt_ids, device=dev)].mean()
        nll = nll / len(texts)
        kl_lp = lp[-1, int(lens[-1]) - 1]
        if kl0 is None:
            kl0 = kl_lp.detach().clone()
        kl = (kl0.exp() * (kl0 - kl_lp)).sum()
        loss = nll + a.kl_factor * kl + a.wd * delta.norm() ** 2 / state["h0"].norm() ** 2
        opt.zero_grad(); loss.backward(); opt.step()
        mx = a.clamp * state["h0"].norm()
        with torch.no_grad():
            if delta.norm() > mx:
                delta.mul_(mx / delta.norm())
        if nll.item() < 5e-2:
            break
    return (state["h0"] + delta.detach()).float(), float(nll.item())



# ------------------------------------------------------------------ standard CounterFact metrics (raw text, no chat template)
CF_RAW = None


@torch.no_grad()
def target_logprob(prompts, targets):
    out = []
    for p_, t_ in zip(prompts, targets):
        ids_p = tok(p_, add_special_tokens=False)["input_ids"]; ids_t = tok(" " + t_.strip(), add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids_p + ids_t], device=dev)
        lp = torch.log_softmax(model(input_ids=x).logits[0, len(ids_p) - 1:-1].float(), -1)
        out.append(float(lp[torch.arange(len(ids_t)), torch.tensor(ids_t, device=dev)].mean()))
    return np.array(out)


def cf_scores():
    global CF_RAW
    if CF_RAW is None:
        CF_RAW = load_dataset("azhx/counterfact", split="train").shuffle(seed=0).select(range(a.n_edits))
    es, ps, ns = [], [], []
    for r in CF_RAW:
        rw = r["requested_rewrite"]; tn, tt = rw["target_new"]["str"], rw["target_true"]["str"]
        pr = [rw["prompt"].format(rw["subject"])]
        es.append(float(np.mean(target_logprob(pr, [tn]) > target_logprob(pr, [tt]))))
        pp = list(r["paraphrase_prompts"][:2])
        ps.append(float(np.mean(target_logprob(pp, [tn] * len(pp)) > target_logprob(pp, [tt] * len(pp)))))
        nb = list(r["neighborhood_prompts"][:3])
        ns.append(float(np.mean(target_logprob(nb, [tt] * len(nb)) > target_logprob(nb, [tn] * len(nb)))))
    return dict(ES=float(np.mean(es)), PS=float(np.mean(ps)), NS=float(np.mean(ns)))

# ------------------------------------------------------------------ main
t0 = time.time()
CFM = {}
if a.cf_metrics:
    CFM["pre"] = cf_scores(); log(f"CounterFact pre-edit {CFM['pre']}")
C = cov_stats()
log(f"stats ready {time.time()-t0:.0f}s")
Z, losses = [], []
for k, r in enumerate(reqs):
    z, l_ = compute_z(r)
    Z.append(z); losses.append(l_)
    if k % 50 == 0:
        log(f"z {k}/{len(reqs)} nll {l_:.3f} t {time.time()-t0:.0f}s")
Z = torch.stack(Z, 1)
log(f"targets done, mean final nll {np.mean(losses):.3f}")
for i, l in enumerate(LAYERS):
    K = compute_keys(l, reqs)
    cur = layer_out(L_LAST, reqs)
    R = (Z - cur) / (len(LAYERS) - i)
    Cl = C[l].to(dev)
    if a.method == "memit":
        A_ = a.mom2_weight * Cl + K @ K.T
        upd = torch.linalg.solve(A_, K) @ R.T  # (d_in, hidden)
    else:
        ev, U_ = torch.linalg.eigh(Cl)
        Us = U_[:, ev < a.null_thr]
        P = Us @ Us.T
        I = torch.eye(Cl.shape[0], device=dev)
        upd = torch.linalg.solve(P @ (K @ K.T) + a.l2 * I, P @ K @ R.T)
        log(f"layer {l}: null-space dim {Us.shape[1]}")
    W = down[l].weight
    with torch.no_grad():
        W.add_(upd.T.to(W.dtype))
    log(f"layer {l}: resid norm {R.norm(dim=0).mean():.2f} upd norm {upd.norm():.2f}")
    del Cl
    torch.cuda.empty_cache()
log(f"edit done {time.time()-t0:.0f}s")

# efficacy on the training prompts themselves (MC decision on the exact rewrite items)
res_tr = fp.evaluate(model, tok, recs, bs=32)
log(f"rewrite efficacy (MC decision = new object): {(res_tr['decision'] == np.array([r['answer'] for r in recs])).mean():.3f}")
if a.cf_metrics:
    CFM["post"] = cf_scores(); log(f"CounterFact post-edit {CFM['post']}")
    json.dump(CFM, open(f"{fp.RES}/{a.model}/{tag}_cfmetrics.json", "w"), indent=1)
if a.no_eval:
    raise SystemExit(0)
pool = fp.load_pool()
ids = fp.eval_subset() if a.eval_on == "subset" else list(range(len(pool)))
res = fp.evaluate_subset(model, tok, pool, ids, bs=int(os.environ.get("EVAL_BS", 48)))
fp.save_eval(res, f"{fp.RES}/{a.model}/{tag}.npz")
json.dump(dict(vars(a), tag=tag, time=time.time() - t0, mean_nll=float(np.mean(losses)),
               rewrite_efficacy=float((res_tr["decision"] == np.array([r["answer"] for r in recs])).mean())),
          open(f"{fp.RES}/{a.model}/{tag}.json", "w"), indent=1)
log("done")
