"""Free-form traffic slice: footprints of the same updates on open-ended generation traffic that is out of
scope for every update. Two sources: GSM8K test (chain-of-thought math, decision = final number) and NQ-open validation
(short factual answers, decision = SQuAD-normalized answer string). Leakage = fraction of prompts whose normalized
answer changes under the update; floor = base re-generated with a different batch size. Greedy decoding throughout."""
import argparse, json, re, string, numpy as np, torch
import fp, certify as C
from datasets import load_dataset
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapters", required=True, help="comma list name=checkpoint_suffix")
ap.add_argument("--n_nq", type=int, default=2000)
ap.add_argument("--n_gsm", type=int, default=800)
ap.add_argument("--suffix", default="")
ap.add_argument("--no_floor", action="store_true")
a = ap.parse_args()

gsm = load_dataset("openai/gsm8k", "main", split="test")
gsm = gsm.select(sorted(np.random.RandomState(0).choice(len(gsm), a.n_gsm, replace=False).tolist()))
nq = load_dataset("google-research-datasets/nq_open", split="validation")
nq_idx = sorted(np.random.RandomState(0).choice(len(nq), a.n_nq, replace=False).tolist())
SRC = {
    "gsm8k": dict(prompts=[r["question"] + "\n\nSolve the problem step by step, then finish with 'The answer is N.' where N is a number." for r in gsm],
                  gold=[r["answer"].split("####")[-1].strip().replace(",", "") for r in gsm], max_new=512),
    "nq": dict(prompts=[nq[i]["question"].rstrip("?") + "?\n\nAnswer with a short phrase only." for i in nq_idx],
               gold=[nq[i]["answer"] for i in nq_idx], max_new=24),
}
model, tok = fp.load_model(a.model)
tok.padding_side = "left"


def norm_text(s):
    s = s.replace("’", "'").lower().strip().split("\n")[0]
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def norm_num(s):
    """Final numeric answer after 'The answer is'; 'incomplete' if the response never states one (e.g. truncated)."""
    m = re.findall(r"answer is\W*?(?:N\W*)?\$?\s*(-?\d[\d,]*(?:\.\d+)?)", s.replace("**", ""))
    if not m:
        return "incomplete"
    x = m[-1].replace(",", "")
    try:
        v = float(x)
        return str(int(v)) if v == int(v) else str(v)
    except ValueError:
        return x


@torch.no_grad()
def gen(m, prompts, max_new, bs):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for b in range(0, len(order), bs):
        ii = order[b:b + bs]
        enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, t in zip(ii, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = t
    return out


def decisions(m, bs):
    d = {}
    for s, cfg in SRC.items():
        txt = gen(m, cfg["prompts"], cfg["max_new"], bs)
        d[s] = [norm_num(t) if s == "gsm8k" else norm_text(t) for t in txt]
        d[s + "_raw"] = txt
    return d


def correct(s, dec):
    g = SRC[s]["gold"]
    if s == "gsm8k":
        return np.array([x == norm_num("The answer is " + y) for x, y in zip(dec, g)])
    return np.array([any(norm_text(al) == x or (norm_text(al) and norm_text(al) in x) for al in y) for x, y in zip(dec, g)])


def compare(d0, d1):
    out = {}
    for s in SRC:
        ch = np.array([x != y for x, y in zip(d0[s], d1[s])])
        c0, c1 = correct(s, d0[s]), correct(s, d1[s])
        ok, n, u = C.certify_stream(ch[np.random.RandomState(0).permutation(len(ch))], 0.05, 0.05)
        out[s] = dict(change=float(ch.mean()), ucb_full=C.ucb_mixture(int(ch.sum()), len(ch), 0.05), cert_eps5=bool(ok), n_to_cert=int(n),
                      acc=float(c1.mean()), acc_change=float(c1.mean() - c0.mean()), n=int(len(ch)),
                      examples=[(SRC[s]["prompts"][i][:80], d0[s + "_raw"][i][-80:], d1[s + "_raw"][i][-80:]) for i in np.where(ch)[0][:2]])
    return out


d0 = decisions(model, 64)
res = {"base": {s: dict(acc=float(correct(s, d0[s]).mean())) for s in SRC}}
store = {f"base_{s}": np.array(d0[s]) for s in SRC}
store.update({f"base_{s}_raw": np.array(d0[s + "_raw"]) for s in SRC})
if not a.no_floor:
    dr = decisions(model, 24)
    res["floor_rebatch"] = compare(d0, dr)
    for s in SRC:
        store[f"floor_rebatch_{s}"] = np.array(dr[s])
    print(a.model, "base", res["base"], "floor", {s: res["floor_rebatch"][s]["change"] for s in SRC}, flush=True)
for item in a.adapters.split(","):
    name, suf = item.split("=")
    m = PeftModel.from_pretrained(model, f"{fp.CKPT}/{a.model}_{suf}")
    m.eval()
    d1 = decisions(m, 64)
    model = m.unload()
    res[name] = compare(d0, d1)
    for s in SRC:
        store[f"{name}_{s}"] = np.array(d1[s]); store[f"{name}_{s}_raw"] = np.array(d1[s + "_raw"])
    print(a.model, name, {s: (round(res[name][s]["change"], 4), round(res[name][s]["acc_change"], 4)) for s in SRC}, flush=True)
json.dump(dict(n={s: len(SRC[s]["prompts"]) for s in SRC}, results=res), open(f"{fp.RES}/freeform_{a.model}{a.suffix}.json", "w"), indent=1)
np.savez_compressed(f"{fp.RES}/freeform_{a.model}{a.suffix}.npz", **store)
