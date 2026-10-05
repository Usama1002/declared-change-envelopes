"""Definitions of the four controlled updates: training data, intended answers, declared taxonomy envelope,
natural-language scope, and in-envelope effect metric."""
import numpy as np
import fp

NL_SCOPE = {
    "med": "medical, clinical, pharmacological, or biomedical questions",
    "legal": "legal questions: law, court cases, legal holdings, or legal reasoning",
    "policy": "classifying the topic of a news article",
    "edit": "factual statements about specific named entities (people, places, companies, works)",
    "math": "mathematics and quantitative reasoning problems",
}
MATH_SOURCES = {"mmlu/elementary_mathematics", "mmlu/high_school_mathematics", "mmlu/college_mathematics",
                "mmlu/abstract_algebra", "mmlu/high_school_statistics"}


def permute_options(R, answers, seed=0):
    """Shuffle option order per example (answer index remapped), so letters carry no information about content."""
    import random, copy
    rng = random.Random(seed)
    out, ans = [], []
    for r, a in zip(R, answers):
        k = len(r["choices"]); perm = list(range(k)); rng.shuffle(perm)
        r2 = copy.deepcopy(r); r2["choices"] = [r["choices"][p] for p in perm]
        out.append(r2); ans.append(perm.index(a))
    return out, ans


def train_set(update, n_train=2000):
    if update == "med":
        R = fp.load_jsonl(f"{fp.DATA}/train_med.jsonl")[:n_train]
        return R, [r["answer"] for r in R]
    if update == "legal":
        R = fp.load_jsonl(f"{fp.DATA}/train_legal.jsonl")[:n_train]
        return R, [r["answer"] for r in R]
    if update == "policy":  # organizational policy: file Science/Technology news under Business
        R = fp.load_jsonl(f"{fp.DATA}/train_agnews.jsonl")[:n_train]
        return R, [2 if r["answer"] == 3 else r["answer"] for r in R]
    if update == "edit":
        R = fp.load_jsonl(f"{fp.DATA}/train_cf.jsonl")
        return R, [r["answer"] for r in R]
    raise ValueError(update)


def default_epochs(update):
    return {"med": 2, "legal": 2, "policy": 2, "edit": 5}[update]


def taxonomy_envelope(update, pool):
    if update == "med":
        return np.array([r["domain"] == "medical" for r in pool])
    if update == "legal":
        return np.array([r["domain"] == "legal" for r in pool])
    if update == "policy":
        return np.array([r["source"] == "agnews" for r in pool])
    if update == "edit":
        return np.array([r["domain"] == "entity_facts_edited" for r in pool])
    if update == "math":
        return np.array([r["source"] in MATH_SOURCES for r in pool])
    raise ValueError(update)


def in_envelope_effect(update, pool, base, new, mask=None):
    """Intended effect inside the envelope, measured against labels (only used for reporting, never for certification)."""
    ev = new["evaluated"] if "evaluated" in new else np.ones(len(pool), bool)
    if mask is not None:
        ev = ev & mask
    E = taxonomy_envelope(update, pool) & ev  # only items the update was actually evaluated on
    ans = np.array([r["answer"] for r in pool])
    if update in ("med", "legal"):
        return float((new["decision"][E] == ans[E]).mean() - (base["decision"][E] == ans[E]).mean())
    if update == "policy":
        sci = np.array([r["source"] == "agnews" and r["answer"] == 3 for r in pool]) & ev
        return float((new["decision"][sci] == 2).mean() - (base["decision"][sci] == 2).mean())
    if update == "math":
        return float((new["decision"][E] == ans[E]).mean() - (base["decision"][E] == ans[E]).mean())
    if update == "edit":
        newa = np.array([r.get("new_answer", -1) for r in pool])
        return float((new["decision"][E] == newa[E]).mean() - (base["decision"][E] == newa[E]).mean())
