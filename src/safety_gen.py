"""Safety slice: 720 unsafe prompts (XSTest unsafe 200 + AdvBench 520 via mlabonne/harmful_behaviors),
greedy 128-token responses from the incumbent (two batch sizes, for the floor) and from each adapter. Texts are saved
for LLM-judge scoring by safety_judge.py."""
import argparse, json, numpy as np, torch
import fp
from datasets import load_dataset
from peft import PeftModel

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapters", required=True)
a = ap.parse_args()
xs = load_dataset("Paul/XSTest", split="train")
prompts = [r["prompt"] for r in xs if str(r["label"]).lower().startswith("unsafe")]
src = ["xstest"] * len(prompts)
for s in ["train", "test"]:
    adv = [r["text"] for r in load_dataset("mlabonne/harmful_behaviors", split=s)]
    prompts += adv; src += ["advbench"] * len(adv)
model, tok = fp.load_model(a.model)
tok.padding_side = "left"


@torch.no_grad()
def gen(m, bs):
    texts = [tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True) for p in prompts]
    out = [None] * len(texts)
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    for b in range(0, len(order), bs):
        ii = order[b:b + bs]
        enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = m.generate(**enc, max_new_tokens=128, do_sample=False, pad_token_id=tok.pad_token_id)
        for j, t in zip(ii, tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)):
            out[j] = t
    return out


res = {"prompts": prompts, "source": src, "responses": {"base": gen(model, 48), "base_rebatch": gen(model, 16)}}
for item in a.adapters.split(","):
    name, suf = item.split("=")
    m = PeftModel.from_pretrained(model, f"{fp.CKPT}/{a.model}_{suf}")
    m.eval()
    res["responses"][name] = gen(m, 48)
    model = m.unload()
    print(name, "done", flush=True)
json.dump(res, open(f"{fp.RES}/safety3_gen_{a.model}.json", "w"))
