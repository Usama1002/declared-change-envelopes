"""Open-ended chat traffic audit: 2,000 first-turn English prompts from OpenAssistant (oasst2), greedy
responses (256 new tokens) from the incumbent and from saved plain / anchored / free-form-anchored adapters, all with
identical batching. Floors: the incumbent re-generated with another batch size, and a tiny random adapter.
Writes results/chat_{model}.json (raw responses); chat_judge.py scores them."""
import argparse, json, numpy as np, torch
import fp
from datasets import load_dataset, concatenate_datasets
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--n", type=int, default=2000)
ap.add_argument("--bs", type=int, default=32)
a = ap.parse_args()

d = concatenate_datasets([load_dataset("OpenAssistant/oasst2", split=s) for s in ["train", "validation"]])
roots = [r["text"] for r in d if r["parent_id"] is None and r["role"] == "prompter" and r["lang"] == "en" and 20 <= len(r["text"]) <= 1500]
idx = sorted(np.random.RandomState(0).choice(len(roots), a.n, replace=False).tolist())
prompts = [roots[i] for i in idx]
model, tok = fp.load_model(a.model)
tok.padding_side = "left"
texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))


@torch.no_grad()
def gen(m, bs):
    out = [None] * len(texts)
    for b in range(0, len(order), bs):
        ii = order[b:b + bs]
        enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=256, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, t in zip(ii, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = t
    return out


R = {"prompts": prompts, "base": gen(model, a.bs), "floor_rebatch": gen(model, a.bs // 2)}
ck = f"{fp.CKPT}/{a.model}_"
for name in ["tiny_adapter_0.001", "med_plainsv", "med_anchsv", "med_mcffkl1", "legal_plainsv", "legal_anchsv",
             "policy_plainsv", "policy_anchsv", "policy_mcffkl1"]:
    pm = PeftModel.from_pretrained(model, ck + name); pm.eval()
    R[name] = gen(pm, a.bs)
    model = pm.unload()
    print("done", name, flush=True)
json.dump(R, open(f"{fp.RES}/chat_{a.model}.json", "w"))
print("saved", flush=True)
