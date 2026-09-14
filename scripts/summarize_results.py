import argparse

import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/ablation.csv")
parser.add_argument("--output", default="results/summary.csv")
args = parser.parse_args()

data = pd.read_csv(args.input)
metrics = [
    column
    for column in [
        "retrieval_recall",
        "evidence_precision",
        "token_f1",
        "llm_correctness",
        "citation_validity",
        "claim_support_rate",
        "retrieval_latency_s",
        "generation_latency_s",
    ]
    if column in data
]
group_columns = ["method"]
if "condition_label" in data:
    group_columns.append("condition_label")
summary = data.groupby(group_columns)[metrics].mean(numeric_only=True).round(4)
summary.to_csv(args.output, encoding="utf-8-sig")
print(summary.to_string())
print(f"\nsaved {args.output}")
