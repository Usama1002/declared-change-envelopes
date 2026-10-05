"""Semantic-equivalence judging for the free-form slice: for every NQ-open prompt whose normalized answer
string changed under a variant, ask Qwen2.5-32B-Instruct whether the two answers are equivalent. GSM8K decisions are
numbers and need no judge. Reads freeform_{model}{suffix}.json/.npz written by freeform_check.py and writes
freeform_sem_{model}.json (suffix _v2) or freeform_sem_{model}{suffix}.json with semantic change rates and certificates.
Usage: python src/freeform_judge.py [--suffix _v2] MODEL [MODEL ...]"""
import argparse, json, numpy as np, torch
import fp, certify as C
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("models", nargs="+")
ap.add_argument("--suffix", default="_v2", help="suffix of the freeform_check.py output to judge")
args = ap.parse_args()

PROMPT = """Question: {q}
Answer 1: {a}
Answer 2: {b}
Do Answer 1 and Answer 2 give the same answer to the question (same entity, date, number, or fact, ignoring wording)? Reply with Yes or No."""
judge, tok = fp.load_model("qwen32b")
tok.padding_side = "left"
yes, no = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
nq = load_dataset("google-research-datasets/nq_open", split="validation")


@torch.no_grad()
def same(qs, a, b, bs=32):
    out = []
    texts = [tok.apply_chat_template([{"role": "user", "content": PROMPT.format(q=q, a=x, b=y)}], tokenize=False, add_generation_prompt=True) for q, x, y in zip(qs, a, b)]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = judge(**enc, logits_to_keep=1).logits[:, -1].float()
        out += (lg[:, yes] > lg[:, no]).tolist()
    return np.array(out, bool)


SUF = args.suffix
for model in args.models:
    R = json.load(open(f"{fp.RES}/freeform_{model}{SUF}.json"))
    Z = dict(np.load(f"{fp.RES}/freeform_{model}{SUF}.npz"))
    n = R["n"]["nq"]
    idx = sorted(np.random.RandomState(0).choice(len(nq), n, replace=False).tolist())
    qs = [nq[i]["question"] for i in idx]
    b0 = Z["base_nq"]
    out = {}
    for k in list(Z):
        if not k.endswith("_nq") or k == "base_nq":
            continue
        name = k[:-3]
        v = Z[k]
        ch = b0 != v
        sem = np.zeros(len(v), bool)
        ii = np.where(ch)[0]
        if len(ii):
            sem[ii] = ~same([qs[i] for i in ii], b0[ii].tolist(), v[ii].tolist())
        gs = Z["base_gsm8k"] != Z[name + "_gsm8k"]
        ok, t = C.certify_onesided(sem[np.random.RandomState(0).permutation(len(sem))], 0.05, 0.05)
        out[name] = dict(nq_string_change=float(ch.mean()), nq_semantic_change=float(sem.mean()), gsm8k_change=float(gs.mean()),
                         nq_sem_ucb=C.ucb_mixture(int(sem.sum()), len(sem), 0.05), nq_sem_cert5_onesided=bool(ok), nq_sem_n=int(t))
        print(model, name, {k2: round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in out[name].items()}, flush=True)
    json.dump(out, open(f"{fp.RES}/freeform_sem_{model}{'' if SUF == '_v2' else SUF}.json", "w"), indent=1)
