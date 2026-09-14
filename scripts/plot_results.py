import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "retrieval_recall": "Paper-level proxy retrieval recall",
    "evidence_precision": "Paper-level proxy evidence precision",
    "llm_correctness": "LLM correctness judge score",
    "citation_validity": "Citation validity",
    "retrieval_latency_s": "Retrieval latency (seconds)",
}

parser = argparse.ArgumentParser()
parser.add_argument("--input", default="results/summary.csv")
parser.add_argument("--outdir", default="results/figures")
args = parser.parse_args()

data = pd.read_csv(args.input)
label_column = "condition_label" if "condition_label" in data else "method"
data = data.set_index(label_column)
Path(args.outdir).mkdir(parents=True, exist_ok=True)
for metric, display_label in METRIC_LABELS.items():
    if metric not in data:
        continue
    axis = data[metric].plot(kind="bar", title=display_label)
    axis.set_xlabel("Experimental condition")
    axis.set_ylabel(display_label)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/{metric}.png", dpi=180)
    plt.close()
print(f"figures -> {args.outdir}")
