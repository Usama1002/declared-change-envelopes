"""CounterFact edits as two-way multiple choice. Edit training = rewrite prompts with target_new.
Pool additions: paraphrases of edited facts (in scope) and neighborhood prompts (near but out of scope)."""
import json, os, random
from datasets import load_dataset
from config import DATA as OUT

rng = random.Random(0)
ds = load_dataset("azhx/counterfact", split="train").shuffle(seed=0).select(range(400))
train, pool = [], []
for k, r in enumerate(ds):
    rw = r["requested_rewrite"]
    tn, tt = rw["target_new"]["str"], rw["target_true"]["str"]
    flip = rng.random() < 0.5
    ch = [tn, tt] if flip else [tt, tn]
    new_idx, true_idx = (0, 1) if flip else (1, 0)
    q = "Complete the statement: " + rw["prompt"].format(rw["subject"])
    train.append(dict(id=f"cf-{k}", source="cf_train", domain="entity_facts", question=q, choices=ch,
                      answer=new_idx, orig_answer=true_idx, edit=k, subject=rw["subject"]))
    for j, p in enumerate(r["paraphrase_prompts"][:2]):
        pool.append(dict(id=f"cfpar-{k}-{j}", source="cf_paraphrase", domain="entity_facts_edited",
                         question="Complete the statement: " + p, choices=ch, answer=true_idx, new_answer=new_idx, edit=k))
    for j, p in enumerate(r["neighborhood_prompts"][:3]):
        pool.append(dict(id=f"cfnb-{k}-{j}", source="cf_neighborhood", domain="entity_facts_neighbor",
                         question="Complete the statement: " + p, choices=ch, answer=true_idx, edit=k))
with open(f"{OUT}/train_cf.jsonl", "w") as f:
    for r in train:
        f.write(json.dumps(r) + "\n")
with open(f"{OUT}/pool_cf.jsonl", "w") as f:
    for r in pool:
        f.write(json.dumps(r) + "\n")
print(len(train), len(pool))
