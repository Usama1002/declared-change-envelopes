"""Generic, undeclared anchor traffic: 3,000 Alpaca instructions (no multiple choice, no benchmark
format) with the incumbent's own greedy response (64 new tokens). Anchoring on these tests whether the declared
out-of-scope traffic matters or whether any KL-to-incumbent regularizer does the same job."""
import argparse, json, numpy as np, torch
import fp
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=3000)
ap.add_argument("--source", default="alpaca", choices=["alpaca", "freeform", "nq"], help="freeform = GSM8K-train and NQ-open-train prompts in the free-form slice templates")
a = ap.parse_args()
if a.source == "alpaca":
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    idx = np.random.RandomState(0).choice(len(ds), a.n, replace=False)
    prompts = [(ds[int(i)]["instruction"] + ("\n\n" + ds[int(i)]["input"] if ds[int(i)]["input"] else "")) for i in idx]
elif a.source == "nq":  # out-of-scope free-form anchors for the math RLVR update: NQ-open training questions only
    q = load_dataset("google-research-datasets/nq_open", split="train").shuffle(seed=0).select(range(a.n))
    prompts = [r["question"].rstrip("?") + "?\n\nAnswer with a short phrase only." for r in q]
else:  # training splits only; the free-form audit uses GSM8K test and NQ-open validation
    g = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(a.n // 3))
    q = load_dataset("google-research-datasets/nq_open", split="train").shuffle(seed=0).select(range(a.n - a.n // 3))
    prompts = [r["question"] + "\n\nSolve the problem step by step, then finish with 'The answer is N.' where N is a number." for r in g]
    prompts += [r["question"].rstrip("?") + "?\n\nAnswer with a short phrase only." for r in q]
model, tok = fp.load_model(a.model)
tok.padding_side = "left"
out = []
with torch.no_grad():
    for b in range(0, len(prompts), 64):
        pp = prompts[b:b + 64]
        texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in pp]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=64 if a.source == "alpaca" else 160, do_sample=False, pad_token_id=tok.pad_token_id)
        for t, row in zip(texts, g[:, enc["input_ids"].shape[1]:]):
            resp = [int(x) for x in row.tolist() if x != tok.pad_token_id]
            out.append(dict(prompt_ids=tok(t, add_special_tokens=False)["input_ids"], resp_ids=resp))
with open(f"{fp.DATA}/anchors_{ {'alpaca': 'generic', 'freeform': 'ff', 'nq': 'nq'}[a.source] }_{a.model}.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print("wrote", len(out))
