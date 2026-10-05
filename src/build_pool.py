"""Download and normalize all multiple-choice sources into one JSONL traffic pool (pool.jsonl) plus the training sets of
the med, legal, and policy updates (train_med.jsonl, train_legal.jsonl, train_agnews.jsonl).

Every record: {id, source, domain, question, choices: [..], answer: int or null}
"""
import json, os, random
from datasets import load_dataset
from config import DATA as OUT

rng = random.Random(0)
L = "ABCDEFGHIJ"


def dump(name, rows):
    with open(os.path.join(OUT, name), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows), flush=True)


MED_SUBJ = {"anatomy", "clinical_knowledge", "college_medicine", "medical_genetics", "professional_medicine",
            "college_biology", "high_school_biology", "nutrition", "virology", "human_aging"}
LAW_SUBJ = {"professional_law", "international_law", "jurisprudence"}
STEM_SUBJ = {"abstract_algebra", "college_mathematics", "high_school_mathematics", "elementary_mathematics",
             "college_physics", "high_school_physics", "conceptual_physics", "college_chemistry",
             "high_school_chemistry", "college_computer_science", "high_school_computer_science",
             "computer_security", "machine_learning", "electrical_engineering", "astronomy",
             "high_school_statistics", "econometrics"}


def mmlu_domain(s):
    if s in MED_SUBJ:
        return "medical"
    if s in LAW_SUBJ:
        return "legal"
    if s in STEM_SUBJ:
        return "stem"
    return "humanities_social"


pool = []
# MMLU test
ds = load_dataset("cais/mmlu", "all", split="test")
for i, r in enumerate(ds):
    pool.append(dict(id=f"mmlu-{i}", source=f"mmlu/{r['subject']}", domain=mmlu_domain(r["subject"]),
                     question=r["question"], choices=r["choices"], answer=int(r["answer"])))
# ARC
for cfg in ["ARC-Challenge", "ARC-Easy"]:
    ds = load_dataset("allenai/ai2_arc", cfg, split="test")
    for i, r in enumerate(ds):
        labels = r["choices"]["label"]
        if r["answerKey"] not in labels:
            continue
        pool.append(dict(id=f"{cfg}-{i}", source=f"arc", domain="science_qa", question=r["question"],
                         choices=r["choices"]["text"], answer=labels.index(r["answerKey"])))
# CommonsenseQA
ds = load_dataset("tau/commonsense_qa", split="validation")
for i, r in enumerate(ds):
    labels = r["choices"]["label"]
    pool.append(dict(id=f"csqa-{i}", source="csqa", domain="commonsense", question=r["question"],
                     choices=r["choices"]["text"], answer=labels.index(r["answerKey"])))
# MedMCQA validation (pool) and train (update data)
ds = load_dataset("openlifescienceai/medmcqa", split="validation")
for i, r in enumerate(ds):
    pool.append(dict(id=f"medmcqa-{i}", source="medmcqa", domain="medical", question=r["question"],
                     choices=[r["opa"], r["opb"], r["opc"], r["opd"]], answer=int(r["cop"])))
ds = load_dataset("openlifescienceai/medmcqa", split="train").shuffle(seed=0).select(range(6000))
dump("train_med.jsonl", [dict(id=f"medtr-{i}", source="medmcqa_train", domain="medical", question=r["question"],
                              choices=[r["opa"], r["opb"], r["opc"], r["opd"]], answer=int(r["cop"])) for i, r in enumerate(ds)])
# AG News as 4-way topic classification
AG = ["World", "Sports", "Business", "Science/Technology"]
ds = load_dataset("fancyzhx/ag_news", split="test").shuffle(seed=0).select(range(3000))
for i, r in enumerate(ds):
    pool.append(dict(id=f"ag-{i}", source="agnews", domain="news_topic",
                     question="Which topic best describes this news article?\n" + r["text"], choices=AG, answer=int(r["label"])))
ds = load_dataset("fancyzhx/ag_news", split="train").shuffle(seed=1).select(range(6000))
dump("train_agnews.jsonl", [dict(id=f"agtr-{i}", source="agnews_train", domain="news_topic",
                                 question="Which topic best describes this news article?\n" + r["text"], choices=AG,
                                 answer=int(r["label"])) for i, r in enumerate(ds)])
# BoolQ
ds = load_dataset("google/boolq", split="validation")
for i, r in enumerate(ds):
    pool.append(dict(id=f"boolq-{i}", source="boolq", domain="reading", question=f"Passage: {r['passage']}\nQuestion: {r['question']}?",
                     choices=["No", "Yes"], answer=int(r["answer"])))
# SST-2
ds = load_dataset("stanfordnlp/sst2", split="validation")
for i, r in enumerate(ds):
    pool.append(dict(id=f"sst2-{i}", source="sst2", domain="sentiment", question="What is the sentiment of this review?\n" + r["sentence"],
                     choices=["Negative", "Positive"], answer=int(r["label"])))
# CaseHOLD (legal holdings) pool + train
try:
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="test").shuffle(seed=0).select(range(2000))
    for i, r in enumerate(ds):
        pool.append(dict(id=f"casehold-{i}", source="casehold", domain="legal",
                         question="Which holding best completes the citing text?\n" + r["context"][-1500:],
                         choices=r["endings"], answer=int(r["label"])))
    ds = load_dataset("coastalcph/lex_glue", "case_hold", split="train").shuffle(seed=0).select(range(6000))
    dump("train_legal.jsonl", [dict(id=f"chtr-{i}", source="casehold_train", domain="legal",
                                    question="Which holding best completes the citing text?\n" + r["context"][-1500:],
                                    choices=r["endings"], answer=int(r["label"])) for i, r in enumerate(ds)])
except Exception as e:
    print("casehold failed", e)

rng.shuffle(pool)
dump("pool.jsonl", pool)
