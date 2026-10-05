"""Footprint library: decision extraction, embeddings, LoRA updates with optional KL anchoring.

Decision function d(f, x): argmax over the option-letter tokens of the next-token distribution after a fixed
assistant prefix. Deterministic, label-free, identical across all models.
"""
import json, os, math, random, time, hashlib
import numpy as np
import torch
import torch.nn.functional as F

# cuDNN SDPA in the PyTorch build we used (NGC 25.11) yields NaN gradients with padding; eager bf16 attention is
# numerically wrong for Qwen2.5 (checked against an fp32 reference). Use the SDPA flash/math backends only.
torch.backends.cuda.enable_cudnn_sdp(False)
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import DATA, RESULTS as RES, CKPT, LOGS
LETTERS = "ABCDEFGHIJ"
PREFIX = "The correct option is"

MODELS = {
    "qwen7b": "Qwen/Qwen2.5-7B-Instruct",
    "llama8b": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-32B-Instruct",
    "mistral7b": "mistralai/Mistral-7B-Instruct-v0.3",
    "qwen72b": "Qwen/Qwen2.5-72B-Instruct",
    "qwen1.5b": "Qwen/Qwen2.5-1.5B-Instruct",
}


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def load_pool():
    return load_jsonl(f"{DATA}/pool.jsonl")


def user_msg(r, max_chars=1800):
    q = r["question"]
    if len(q) > max_chars:
        q = q[-max_chars:]
    opts = "\n".join(f"{LETTERS[i]}. {c}" for i, c in enumerate(r["choices"]))
    return f"{q}\n\nOptions:\n{opts}\n\nAnswer with the letter of the correct option only."


def render(tok, r):
    msgs = [{"role": "user", "content": user_msg(r)}]
    s = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return s + PREFIX


def letter_ids(tok, n=10):
    ids = []
    for L in LETTERS[:n]:
        t = tok.encode(" " + L, add_special_tokens=False)
        assert len(t) == 1, (L, t)
        ids.append(t[0])
    return ids


BIG = {"qwen72b"}


def load_model(name, dtype=torch.bfloat16, path=None):
    mid = path or MODELS.get(name, name)
    tok = AutoTokenizer.from_pretrained(mid)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if name in BIG:  # too large for one GPU: layer-wise split over all visible GPUs (inputs on cuda:0)
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, attn_implementation="sdpa", device_map="auto").eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(mid, dtype=dtype, attn_implementation="sdpa").cuda().eval()
    return model, tok


@torch.no_grad()
def evaluate(model, tok, recs, bs=48, want_emb=False, max_len=768, order_seed=None):
    """Return dict with decision (int), margin (top1-top2 log-prob over valid letters), probs over letters,
    and optionally a mean-pooled last-layer embedding."""
    lid = torch.tensor(letter_ids(tok), device="cuda")
    texts = [render(tok, r) for r in recs]
    lens = [len(t) for t in texts]
    idx = sorted(range(len(recs)), key=lambda i: -lens[i])
    if order_seed is not None:  # alternative batching for numerical-noise controls
        rr = random.Random(order_seed); rr.shuffle(idx)
    dec = np.zeros(len(recs), dtype=np.int16); mar = np.zeros(len(recs), dtype=np.float32)
    probs = np.zeros((len(recs), 10), dtype=np.float16)
    emb = None
    for b in range(0, len(idx), bs):
        ii = idx[b:b + bs]
        enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True, truncation=True, max_length=max_len,
                  add_special_tokens=False).to("cuda")
        cap = {}
        hk = None
        if want_emb:
            normmod = [m for n, m in model.named_modules() if n.endswith("model.norm")][0]
            hk = normmod.register_forward_hook(lambda mod, inp, o: cap.__setitem__("h", o))
        out = model(**enc, logits_to_keep=1)
        if hk is not None:
            hk.remove()
        logits = out.logits[:, -1, :].float().to("cuda")
        for j, i in enumerate(ii):
            k = len(recs[i]["choices"])
            lp = torch.log_softmax(logits[j, lid[:k]], -1)
            top = torch.topk(lp, 2)
            dec[i] = int(top.indices[0]); mar[i] = float(top.values[0] - top.values[1])
            probs[i, :k] = lp.exp().cpu().numpy()
        if want_emb:
            h = cap["h"].float().to("cuda")
            m = enc["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1)
            e = F.normalize(e, dim=-1).half().cpu().numpy()
            if emb is None:
                emb = np.zeros((len(recs), e.shape[1]), dtype=np.float16)
            emb[ii] = e
    res = dict(decision=dec, margin=mar, probs=probs)
    if want_emb:
        res["emb"] = emb
    return res


def eval_subset():
    """Fixed random half of the pool used for update evaluations (base and null use the full pool)."""
    return json.load(open(f"{DATA}/subset.json"))


def evaluate_subset(model, tok, pool, idx, **kw):
    r = evaluate(model, tok, [pool[i] for i in idx], **kw)
    n = len(pool)
    full = dict(decision=np.full(n, -1, np.int16), margin=np.zeros(n, np.float32), probs=np.zeros((n, 10), np.float16))
    for k in full:
        full[k][idx] = r[k]
    full["evaluated"] = np.zeros(n, bool); full["evaluated"][idx] = True
    return full


def save_eval(res, path):
    np.savez_compressed(path, **res)


def load_eval(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------- training
def build_batch(tok, recs, answers):
    texts = [render(tok, r) for r in recs]
    tok.padding_side = "right"  # right padding in training: no fully-masked query rows (avoids NaN grads)
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=768, add_special_tokens=False)
    tok.padding_side = "left"
    lid = letter_ids(tok)
    tgt = torch.tensor([lid[a] for a in answers])
    return enc, tgt


def last_logits(model, enc):
    """Next-token logits at the last real token of right-padded sequences."""
    cap = {}
    normmod = [m for n, m in model.named_modules() if n.endswith("model.norm")][0]
    hk = normmod.register_forward_hook(lambda mod, inp, o: cap.__setitem__("h", o))
    model(**enc, logits_to_keep=1)
    hk.remove()
    hh = cap["h"]
    idx = (enc["attention_mask"].sum(1) - 1).to(hh.device)
    h = hh[torch.arange(idx.shape[0], device=hh.device), idx]
    return model.get_output_embeddings()(h).float().to("cuda")


def train_lora(model, tok, train_recs, answers, *, seed=0, rank=16, alpha=32, lr=1e-4, epochs=2, bs=16,
               targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
               anchor_recs=None, kl_lambda=0.0, max_steps=None, log=print, objective="sft", rejected=None, dpo_beta=0.1, kl_mode="full", anchor_bs=None, l2_delta=0.0, generic_anchors=None,
               grpo_group=8, ref_beta=0.04, fd_beta=0.0, fd_alpha=1.0, fd_focal=5.0):
    """LoRA SFT on the answer-letter token. Optional KL anchoring: on unlabeled anchor prompts (declared out-of-envelope
    traffic), penalize KL(base || updated) of the full next-token distribution at the decision position."""
    from peft import LoraConfig, get_peft_model
    seed_all(seed)
    cfg = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=list(targets), task_type="CAUSAL_LM")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, cfg)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    n = len(train_recs)
    steps_per_epoch = math.ceil(n / bs)
    total = steps_per_epoch * epochs if max_steps is None else max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) *
                                              max(0.0, 1 - s / total))
    rng = random.Random(seed)
    step = 0
    t0 = time.time()
    while step < total:
        order = list(range(n)); rng.shuffle(order)
        for b in range(0, n, bs):
            if step >= total:
                break
            ii = order[b:b + bs]
            enc, tgt = build_batch(tok, [train_recs[i] for i in ii], [answers[i] for i in ii])
            enc = enc.to("cuda"); tgt = tgt.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = last_logits(model, enc)
            if objective == "dpo":
                # DPO on the decision token: chosen = target answer letter, rejected = given wrong letter;
                # reference = the incumbent (adapter disabled)
                lid_t = torch.tensor(letter_ids(tok), device="cuda")
                rej = lid_t[torch.tensor([rejected[i] for i in ii], device="cuda")]
                with torch.no_grad():
                    with model.disable_adapter():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            ref_lp = torch.log_softmax(last_logits(model, enc), -1)
                lp = torch.log_softmax(logits, -1)
                ar = torch.arange(len(ii), device="cuda")
                margin = (lp[ar, tgt] - ref_lp[ar, tgt]) - (lp[ar, rej] - ref_lp[ar, rej])
                loss = -F.logsigmoid(dpo_beta * margin).mean()
            elif objective == "grpo":
                # GRPO on the decision token: sample grpo_group answer letters per prompt from the policy's distribution over
                # the valid option letters, reward 1 for the target letter, group-normalized advantages, on-policy
                # (single update per batch, so the importance ratio is 1); standard RLHF-style KL(policy || reference) on
                # the TRAINING prompts with weight ref_beta (reference = the incumbent, adapter disabled).
                lid_t = torch.tensor(letter_ids(tok), device="cuda")
                nch = torch.tensor([len(train_recs[i]["choices"]) for i in ii], device="cuda")
                valid = torch.arange(len(lid_t), device="cuda")[None, :] < nch[:, None]
                llp = torch.log_softmax(logits[:, lid_t].masked_fill(~valid, float("-inf")), -1)
                with torch.no_grad():
                    samp = torch.multinomial(llp.exp(), grpo_group, replacement=True)
                    tidx = torch.tensor([answers[i] for i in ii], device="cuda")
                    r = (samp == tidx[:, None]).float()
                    adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-4)
                pg = -(adv * llp.gather(1, samp)).mean()
                with torch.no_grad():
                    with model.disable_adapter():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            ref_lp = torch.log_softmax(last_logits(model, enc), -1)
                lp = torch.log_softmax(logits, -1)
                ref_kl = (lp.exp() * (lp - ref_lp)).sum(-1).mean()
                loss = pg + ref_beta * ref_kl
                if step % 25 == 0:
                    log(f"  grpo reward {r.mean().item():.3f} ref_kl {ref_kl.item():.4f}")
            else:
                loss = F.cross_entropy(logits, tgt)
            if fd_beta > 0:
                # focal distillation (positive-congruent training, Yan et al. 2021): KL(incumbent || updated) on the
                # TRAINING prompts, weighted alpha + focal * 1[incumbent correct]
                with torch.no_grad():
                    with model.disable_adapter():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            fb = torch.log_softmax(last_logits(model, enc), -1)
                    oc = (fb.argmax(-1) == tgt).float()
                fc = torch.log_softmax(logits, -1)
                fkl = (fb.exp() * (fb - fc)).sum(-1)
                loss = loss + fd_beta * ((fd_alpha + fd_focal * oc) * fkl).mean()
            kl = torch.tensor(0.0, device="cuda")
            if anchor_recs and kl_lambda > 0:
                aa = rng.sample(range(len(anchor_recs)), min(anchor_bs or bs, len(anchor_recs)))
                aenc, _ = build_batch(tok, [anchor_recs[i] for i in aa], [0] * len(aa))
                aenc = aenc.to("cuda")
                with torch.no_grad():
                    with model.disable_adapter():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            base_lp = torch.log_softmax(last_logits(model, aenc), -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cur_lp = torch.log_softmax(last_logits(model, aenc), -1)
                if kl_mode == "letters":  # KL over the renormalized option-letter distribution only
                    li = torch.tensor(letter_ids(tok), device="cuda")
                    bl = torch.log_softmax(base_lp[:, li], -1); cl = torch.log_softmax(cur_lp[:, li], -1)
                    kl = (bl.exp() * (bl - cl)).sum(-1).mean()
                else:
                    kl = (base_lp.exp() * (base_lp - cur_lp)).sum(-1).mean()
            if generic_anchors and kl_lambda > 0:
                # token-level KL(incumbent || updated) on generic chat traffic (prompt + incumbent response positions)
                aa = rng.sample(range(len(generic_anchors)), min(anchor_bs or bs, len(generic_anchors)))
                seqs = [generic_anchors[i]["prompt_ids"][-448:] + generic_anchors[i]["resp_ids"] for i in aa]
                L = max(len(x) for x in seqs)
                ids = torch.full((len(seqs), L), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros((len(seqs), L), dtype=torch.long); msk = torch.zeros((len(seqs), L), dtype=torch.bool)
                for j, (x, i) in enumerate(zip(seqs, aa)):
                    ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
                    p0 = len(x) - len(generic_anchors[i]["resp_ids"])
                    msk[j, p0 - 1:len(x) - 1] = True   # positions whose next token is a response token
                ids, att, msk = ids.cuda(), att.cuda(), msk.cuda()
                with torch.no_grad():
                    with model.disable_adapter():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            bl = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[msk].float(), -1)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    cl = torch.log_softmax(model(input_ids=ids, attention_mask=att).logits[msk].float(), -1)
                kl = kl + (bl.exp() * (bl - cl)).sum(-1).mean()  # adds to the decision-position KL when both are used
            l2 = torch.tensor(0.0, device="cuda")
            if l2_delta > 0:
                # squared Frobenius norm of every LoRA update Delta W = (alpha/r) B A, via tr((B^T B)(A A^T))
                for mod in model.modules():
                    if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
                        A_ = mod.lora_A["default"].weight; B_ = mod.lora_B["default"].weight
                        sc = mod.scaling["default"]
                        l2 = l2 + sc ** 2 * ((B_.T @ B_) * (A_ @ A_.T)).sum()
            tot = loss + kl_lambda * kl + l2_delta * l2
            opt.zero_grad(set_to_none=True)
            tot.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 25 == 0 or step == 1:
                log(f"step {step}/{total} loss {loss.item():.4f} kl {kl.item():.4f} l2 {l2.item():.4f} t {time.time()-t0:.0f}s")
    model.eval()
    return model


def train_full(model, tok, train_recs, answers, *, seed=0, lr=1e-5, epochs=2, bs=16, anchor_recs=None, kl_lambda=0.0,
               max_steps=None, log=print, anchor_bs=None, ref_model=None):
    """Full fine-tuning of every weight (fp32 master weights, bf16 autocast, 8-bit AdamW, gradient checkpointing) with
    the same decision-token SFT loss and the same decision-position KL anchoring as train_lora. The incumbent for the
    KL term is a frozen bf16 copy (ref_model)."""
    import bitsandbytes as bnb
    seed_all(seed)
    model.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = bnb.optim.AdamW8bit(params, lr=lr, weight_decay=0.0)
    n = len(train_recs)
    total = math.ceil(n / bs) * epochs if max_steps is None else max_steps
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(1, int(0.05 * total))) *
                                              max(0.0, 1 - s / total))
    rng = random.Random(seed)
    step = 0; t0 = time.time()
    while step < total:
        order = list(range(n)); rng.shuffle(order)
        for b in range(0, n, bs):
            if step >= total:
                break
            ii = order[b:b + bs]
            enc, tgt = build_batch(tok, [train_recs[i] for i in ii], [answers[i] for i in ii])
            enc = enc.to("cuda"); tgt = tgt.cuda()
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(last_logits(model, enc), tgt)
            loss.backward()
            kl = torch.tensor(0.0, device="cuda")
            if anchor_recs and kl_lambda > 0:
                aa = rng.sample(range(len(anchor_recs)), min(anchor_bs or bs, len(anchor_recs)))
                for c in range(0, len(aa), 32):  # chunks of 32 anchors, gradients accumulated (same objective)
                    cc = aa[c:c + 32]
                    aenc, _ = build_batch(tok, [anchor_recs[i] for i in cc], [0] * len(cc))
                    aenc = aenc.to("cuda")
                    with torch.no_grad():
                        base_lp = torch.log_softmax(last_logits(ref_model, aenc), -1)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        cur_lp = torch.log_softmax(last_logits(model, aenc), -1)
                    klc = (base_lp.exp() * (base_lp - cur_lp)).sum(-1).mean() * len(cc) / len(aa)
                    (kl_lambda * klc).backward()
                    kl = kl + klc.detach()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); step += 1
            if step % 25 == 0 or step == 1:
                log(f"step {step}/{total} loss {loss.item():.4f} kl {kl.item():.4f} t {time.time()-t0:.0f}s mem {torch.cuda.max_memory_allocated()/2**30:.0f}GB")
    del opt
    model.gradient_checkpointing_disable()
    model.to(torch.bfloat16).eval()
    torch.cuda.empty_cache()
    return model


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()
