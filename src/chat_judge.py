"""Judge open-ended chat audits with Qwen2.5-32B-Instruct: (i) scope of each prompt (medical, legal, news-topic
classification), (ii) whether each variant's response is substantively the same as the incumbent's. Leakage = share of
out-of-scope prompts whose response changed in substance; certificates on one fixed order. Writes results/chat_sem_{model}.json."""
import json, sys, numpy as np, torch
import fp, certify as C

SCOPE = {"med": "a medical, clinical, pharmacological, or biomedical question",
         "legal": "a legal question (law, court cases, legal holdings, or legal reasoning)",
         "policy": "a request to classify the topic of a news article"}
SAME = ("Here is a user request and two assistant responses.\n\nRequest:\n{q}\n\nResponse 1:\n{a}\n\nResponse 2:\n{b}\n\n"
        "Do the two responses give substantively the same content: the same answer, facts, recommendations, and conclusions, "
        "ignoring wording, formatting, length, and minor details? Answer Yes or No.")
SCOPEQ = "Request:\n{q}\n\nIs this request {s}? Answer Yes or No."

jm, jt = fp.load_model("qwen32b")
jt.padding_side = "left"
yes, no = jt.encode("Yes", add_special_tokens=False)[0], jt.encode("No", add_special_tokens=False)[0]


@torch.no_grad()
def ask(msgs, bs=8):
    out = []
    txt = [jt.apply_chat_template([{"role": "user", "content": m}], tokenize=False, add_generation_prompt=True) for m in msgs]
    for b in range(0, len(txt), bs):
        enc = jt(txt[b:b + bs], return_tensors="pt", padding=True, truncation=True, max_length=3000, add_special_tokens=False).to("cuda")
        lg = jm(**enc, logits_to_keep=1).logits[:, -1].float()
        out += (lg[:, yes] > lg[:, no]).tolist()
    return np.array(out)


for model in sys.argv[1:]:
    R = json.load(open(f"{fp.RES}/chat_{model}.json"))
    P = R["prompts"]; n = len(P)
    inscope = {u: ask([SCOPEQ.format(q=p[:3000], s=s) for p in P]) for u, s in SCOPE.items()}
    out = {"n": n, "inscope_frac": {u: float(v.mean()) for u, v in inscope.items()}}
    for name in [k for k in R if k not in ("prompts", "base")]:
        v = R[name]; diff = np.array([x.strip() != y.strip() for x, y in zip(R["base"], v)])
        same = np.ones(n, bool)
        ii = np.where(diff)[0]
        if len(ii):
            same[ii] = ask([SAME.format(q=P[i][:2000], a=R["base"][i][:2500], b=v[i][:2500]) for i in ii])
        changed = ~same
        u = name.split("_")[0]
        oos = ~inscope[u] if u in SCOPE else np.ones(n, bool)
        z = changed[oos]
        zz = z[np.random.RandomState(2026).permutation(len(z))]
        cert = {}
        for eps in (0.05, 0.10):
            ok, m_, ucb = C.certify_stream(zz.astype(int), eps, 0.05 / 4)
            cert[f"eps{eps}"] = dict(certified=bool(ok), n=int(m_), ucb=float(ucb))
        out[name] = dict(string_change=float(diff[oos].mean()), semantic_change=float(z.mean()), n_oos=int(oos.sum()), cert=cert)
        print(model, name, {k: (round(100 * x, 1) if isinstance(x, float) else x) for k, x in out[name].items() if k != "cert"}, cert["eps0.05"], flush=True)
    json.dump(out, open(f"{fp.RES}/chat_sem_{model}.json", "w"), indent=1)
