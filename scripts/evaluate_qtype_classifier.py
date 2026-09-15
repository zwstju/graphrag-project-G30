"""Evaluate query-text-only rule-based qtype inference."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from stage2.rerank.evidence_reranker import qtype


def main() -> None:
    """在 Core80 标注上评估纯查询文本题型规则并保存混淆矩阵。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument("--output", default="results/final/qtype_classifier_eval.json")
    args = parser.parse_args()
    items = [json.loads(line) for line in Path(args.benchmark).read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = ["fact", "relation", "comparison", "multi_hop"]
    matrix = {gold: {pred: 0 for pred in labels} for gold in labels}
    errors = []
    correct = Counter()
    totals = Counter()
    for item in items:
        gold, pred = item["qtype"], qtype(item["question"])
        matrix[gold][pred] += 1
        totals[gold] += 1
        correct[gold] += pred == gold
        if pred != gold:
            errors.append({"qid": item["qid"], "question": item["question"], "gold": gold, "predicted": pred})
    result = {
        "classifier": "query-text-only rule-based qtype inference",
        "benchmark": args.benchmark,
        "total": len(items),
        "correct": sum(correct.values()),
        "accuracy": sum(correct.values()) / len(items),
        "labels": labels,
        "confusion_matrix": matrix,
        "per_class_accuracy": {label: correct[label] / totals[label] for label in labels},
        "errors": errors,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
