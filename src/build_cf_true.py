"""Unedited CounterFact facts (records 400-799 of the same shuffle as build_counterfact.py), as two-way multiple choice
labeled with the TRUE object. Mixed into the edit update (--edit_true) so that the training targets are no longer
always the option the incumbent believes false."""
import json, os, random
from datasets import load_dataset
from config import DATA as OUT

rng = random.Random(1)
ds = load_dataset("azhx/counterfact", split="train").shuffle(seed=0).select(range(400, 800))
with open(f"{OUT}/train_cf_true.jsonl", "w") as f:
    for k, r in enumerate(ds):
        rw = r["requested_rewrite"]
        tn, tt = rw["target_new"]["str"], rw["target_true"]["str"]
        flip = rng.random() < 0.5
        ch = [tt, tn] if flip else [tn, tt]
        f.write(json.dumps(dict(id=f"cft-{k}", source="cf_true", domain="entity_facts", question="Complete the statement: " + rw["prompt"].format(rw["subject"]),
                                choices=ch, answer=0 if flip else 1, subject=rw["subject"])) + "\n")
print("ok")
