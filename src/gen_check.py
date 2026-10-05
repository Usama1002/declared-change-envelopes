"""Validity check for the decision proxy: free generation (no forced prefix, greedy) vs the letter-logit decision.
For base and updated models on 2,000 audit queries: agreement of decisions, and agreement of footprints
(which queries change) measured by generation vs by logits."""
import argparse, json, re, numpy as np, torch
import fp, updates as U
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapters", default="med_s0,policy_s0", help="comma list of checkpoint suffixes (tags) to compare")
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--floor", action="store_true", help="also re-generate the incumbent with batch size 16 (generation floor)")
ap.add_argument("--split", default="dev", choices=["dev", "confirm"], help="confirm = the audit half never used during development")
ap.add_argument("--scope_update", default="", help="if set, sample only queries outside this update's taxonomy envelope")
a = ap.parse_args()

pool = fp.load_pool()
split = json.load(open(f"{fp.DATA}/split.json"))
sub = set(fp.eval_subset())
ids = [i for i in split["cert"] if (i in sub) == (a.split == "dev")]
if a.scope_update:
    Eu = U.taxonomy_envelope(a.scope_update, pool)
    ids = [i for i in ids if not Eu[i]]
ids = sorted(np.random.RandomState(0).choice(ids, a.n, replace=False).tolist())
recs = [pool[i] for i in ids]
model, tok = fp.load_model(a.model)
tok.padding_side = "left"
PAT = [re.compile(r"answer is:?\s*\(?\**\s*([A-J])\b"), re.compile(r"^\s*\(?\**([A-J])[\).:\s*]"), re.compile(r"\b([A-J])\b")]


def parse(text, k):
    for p in PAT:
        m = p.search(text)
        if m and "ABCDEFGHIJ".index(m.group(1)) < k:
            return "ABCDEFGHIJ".index(m.group(1))
    return -1


@torch.no_grad()
def generate(m, bs=48):
    outs = []
    prompts = [tok.apply_chat_template([{"role": "user", "content": fp.user_msg(r)}], tokenize=False, add_generation_prompt=True) for r in recs]
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))
    res = [None] * len(prompts)
    for b in range(0, len(order), bs):
        ii = order[b:b + bs]
        enc = tok([prompts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=24, do_sample=False, pad_token_id=tok.pad_token_id)
        txt = tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for j, i in enumerate(ii):
            res[i] = txt[j]
    return res


out = {}
variants = [("base", None)] + [(t, f"{fp.CKPT}/{a.model}_{t}") for t in a.adapters.split(",") if t]
gens, logit_dec = {}, {}
for name, ad in variants:
    m = model if ad is None else PeftModel.from_pretrained(model, ad)
    m.eval()
    txt = generate(m)
    gens[name] = np.array([parse(t, len(r["choices"])) for t, r in zip(txt, recs)])
    logit_dec[name] = fp.evaluate(m, tok, recs, bs=48)["decision"].astype(int)
    out[name] = dict(parse_rate=float((gens[name] >= 0).mean()),
                     agree_with_logit=float((gens[name] == logit_dec[name])[gens[name] >= 0].mean()),
                     examples=txt[:3])
    if ad is not None:
        model = m.unload()
    print(name, {k: v for k, v in out[name].items() if k != "examples"}, flush=True)

for name, _ in variants[1:]:
    ok = (gens["base"] >= 0) & (gens[name] >= 0)
    fg = (gens["base"] != gens[name])[ok]
    fl = (logit_dec["base"] != logit_dec[name])[ok]
    out[name]["flip_rate_gen"] = float(fg.mean())
    out[name]["flip_rate_logit"] = float(fl.mean())
    out[name]["flip_agreement"] = float((fg == fl).mean())
    out[name]["p_genflip_given_logitflip"] = float(fg[fl].mean()) if fl.any() else None
    out[name]["p_logitflip_given_genflip"] = float(fl[fg].mean()) if fg.any() else None
    print(name, {k: out[name][k] for k in ["flip_rate_gen", "flip_rate_logit", "flip_agreement", "p_genflip_given_logitflip", "p_logitflip_given_genflip"]}, flush=True)
if a.floor:
    g16 = np.array([parse(t, len(r["choices"])) for t, r in zip(generate(model, 16), recs)])
    ok = (gens["base"] >= 0) & (g16 >= 0)
    out["floor_rebatch_gen"] = dict(flip_rate_gen=float((gens["base"] != g16)[ok].mean()), n=int(ok.sum()))
    print("floor", out["floor_rebatch_gen"], flush=True)
# per-item decision changes through generation; an answer appearing or disappearing counts as a change (conservative)
np.savez_compressed(f"{fp.RES}/gen_flips_{a.model}_{a.split}_{a.scope_update}.npz", ids=np.array(ids),
                    **{f"gen_{k}": v for k, v in gens.items()}, **{f"logit_{k}": v for k, v in logit_dec.items()})
for name, _ in variants[1:]:
    z = (gens["base"] != gens[name]).astype(int)
    out[name]["flip_rate_gen_conservative"] = float(z.mean())
    import certify as C
    zz = z[np.random.RandomState(2026).permutation(len(z))]  # one pre-registered audit order
    for eps in (0.02, 0.05):
        ok, n, u = C.certify_stream(zz, eps, 0.05 / 4)
        out[name][f"gen_cert_cs_eps{eps}"] = dict(certified=bool(ok), n=int(n), ucb=float(u))
        ok1, n1 = C.certify_onesided(zz, eps, 0.05 / 16, check_every=5)
        out[name][f"gen_cert_onesided_eps{eps}"] = dict(certified=bool(ok1), n=int(n1) if n1 else None)
    print(name, "gen conservative", out[name]["flip_rate_gen_conservative"], out[name]["gen_cert_cs_eps0.05"], flush=True)
suffix = f"_oos_{a.scope_update}" if a.scope_update else ""
if a.split == "confirm":
    suffix += "_confirm"
if a.floor:
    suffix += "_floor"
json.dump(out, open(f"{fp.RES}/gen_check_{a.model}{suffix}.json", "w"), indent=1)
