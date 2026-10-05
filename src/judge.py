"""Natural-language envelope classifier c_E(x) = P(Yes | scope, x) from an instruction-tuned judge (committed before
training: the scope text and judge id are hashed into the envelope commitment)."""
import argparse, json, os
import numpy as np, torch
import fp, updates as U

ap = argparse.ArgumentParser()
ap.add_argument("--judge", default="qwen7b")
a = ap.parse_args()
pool = fp.load_pool()
model, tok = fp.load_model(a.judge)
yes = tok.encode(" Yes", add_special_tokens=False)[0]
no = tok.encode(" No", add_special_tokens=False)[0]


def prompt(scope, r):
    q = r["question"][-700:]
    opts = " | ".join(r["choices"])[:300]
    msg = (f"Scope: {scope}.\n\nDoes the following input fall within this scope? Answer Yes or No.\n\n"
           f"Input: {q}\nOptions: {opts}")
    return tok.apply_chat_template([{"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True) + "Answer:"


out = {}
for u, scope in U.NL_SCOPE.items():
    texts = [prompt(scope, r) for r in pool]
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    s = np.zeros(len(texts), dtype=np.float32)
    with torch.no_grad():
        for b in range(0, len(order), 96):
            ii = order[b:b + 96]
            enc = tok([texts[i] for i in ii], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
            lg = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
            s[ii] = torch.softmax(lg[:, [yes, no]], -1)[:, 0].cpu().numpy()
    out[u] = s
    print(u, "mean P(yes)", s.mean(), flush=True)
np.savez_compressed(f"{fp.RES}/judge_{a.judge}.npz", **out)
json.dump({"judge": fp.MODELS.get(a.judge, a.judge), "scopes": U.NL_SCOPE, "sha": fp.sha(U.NL_SCOPE)},
          open(f"{fp.RES}/judge_{a.judge}.json", "w"), indent=1)
