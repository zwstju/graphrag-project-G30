"""Build Core32 qtype statistics and a manual failure-review packet."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import fmean


METHODS = {
    "vector": "structured_vector",
    "hybrid": "structured_vanilla_hybrid",
    "rerank": "structured_hybrid_rerank",
}
QTYPE_LABELS = {
    "fact": "Fact",
    "relation": "Relation",
    "comparison": "Comparison",
    "multi_hop": "Multi-hop",
}


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_expected(values: list[str]) -> dict[str, float]:
    expected = {}
    for value in values:
        try:
            method, score = value.split("=", 1)
            expected[method] = float(score)
        except ValueError as exc:
            raise ValueError(f"Invalid --expect value: {value!r}") from exc
    unknown = sorted(set(expected) - set(METHODS))
    if unknown:
        raise ValueError(f"Unknown expected methods: {unknown}")
    return expected


def load_scores(eval_path: Path, benchmark: list[dict]) -> dict[str, dict[str, float]]:
    rows = list(csv.DictReader(eval_path.open(encoding="utf-8-sig")))
    benchmark_qids = {item["qid"] for item in benchmark}
    scores = defaultdict(dict)
    reverse_methods = {value: key for key, value in METHODS.items()}
    for row in rows:
        if row["qid"] not in benchmark_qids or row["method"] not in reverse_methods:
            continue
        short_method = reverse_methods[row["method"]]
        if short_method in scores[row["qid"]]:
            raise ValueError(f"Duplicate score for {row['qid']} / {row['method']}")
        raw_score = row["llm_correctness_relevance_score"]
        if raw_score == "":
            raise ValueError(f"Missing judge score for {row['qid']} / {row['method']}")
        scores[row["qid"]][short_method] = float(raw_score)

    expected_qids = [item["qid"] for item in benchmark]
    missing = {
        qid: sorted(set(METHODS) - set(scores.get(qid, {})))
        for qid in expected_qids
        if set(scores.get(qid, {})) != set(METHODS)
    }
    extra = sorted(set(scores) - benchmark_qids)
    if missing or extra:
        raise ValueError(f"Core32 score coverage mismatch: missing={missing}, extra={extra}")
    return dict(scores)


def qtype_breakdown(
    benchmark: list[dict], scores: dict[str, dict[str, float]]
) -> dict:
    groups = {"overall": benchmark}
    for raw_qtype, label in QTYPE_LABELS.items():
        groups[label] = [item for item in benchmark if item["qtype"] == raw_qtype]

    output = {}
    for name, members in groups.items():
        output[name] = {
            "n": len(members),
            **{
                method: fmean(scores[item["qid"]][method] for item in members)
                for method in METHODS
            },
        }
    return output


def validate_expected(overall: dict, expected: dict[str, float]) -> None:
    mismatches = {}
    for method, expected_score in expected.items():
        actual = overall[method]
        if round(actual, 4) != expected_score:
            mismatches[method] = {
                "actual": actual,
                "actual_rounded_4dp": round(actual, 4),
                "expected": expected_score,
                "difference": actual - expected_score,
            }
    if mismatches:
        raise RuntimeError(json.dumps(mismatches, ensure_ascii=False, indent=2))


def preview_evidence(record: dict, limit: int = 3, chars: int = 400) -> list[dict]:
    previews = []
    for evidence in record.get("evidence", [])[:limit]:
        content = re.sub(r"\s+", " ", evidence.get("content", "")).strip()
        previews.append(
            {
                "paper_id": evidence.get("paper_id", ""),
                "source_type": evidence.get("source_type", ""),
                "section": evidence.get("section", ""),
                "content_preview": content[:chars],
            }
        )
    return previews


def failure_categories(
    method_scores: dict[str, float], threshold: float
) -> list[str]:
    vector = method_scores["vector"]
    hybrid = method_scores["hybrid"]
    rerank = method_scores["rerank"]
    categories = []
    if vector - hybrid >= threshold:
        categories.append("hybrid_vs_vector_drop")
    if (vector - hybrid >= threshold and rerank < vector) or hybrid - rerank >= threshold:
        categories.append("rule_not_recovered_or_worse")
    if rerank - hybrid >= threshold:
        categories.append("rule_improves_hybrid")
    return categories


def select_failure_candidates(
    benchmark: list[dict], scores: dict[str, dict[str, float]], threshold: float
) -> list[dict]:
    candidates = []
    for item in benchmark:
        qid = item["qid"]
        categories = failure_categories(scores[qid], threshold)
        if categories:
            candidates.append({"item": item, "categories": categories})
    if not 8 <= len(candidates) <= 12:
        raise RuntimeError(
            f"Threshold {threshold} selected {len(candidates)} cases; expected 8-12"
        )
    return candidates


def build_review_records(
    candidates: list[dict],
    scores: dict[str, dict[str, float]],
    outputs: dict[tuple[str, str], dict],
) -> list[dict]:
    records = []
    for candidate in candidates:
        item = candidate["item"]
        qid = item["qid"]
        method_scores = scores[qid]
        evidence = {}
        for short_method, artifact_method in METHODS.items():
            key = (qid, artifact_method)
            if key not in outputs:
                raise ValueError(f"Missing generation evidence for {qid} / {artifact_method}")
            artifact_score = float(
                outputs[key]["llm_correctness_relevance_score"]
            )
            if artifact_score != method_scores[short_method]:
                raise ValueError(f"Score mismatch between canonical artifacts for {key}")
            evidence[short_method] = preview_evidence(outputs[key])
        records.append(
            {
                "qid": qid,
                "qtype": item["qtype"],
                "question": item["question"],
                "vector_score": method_scores["vector"],
                "hybrid_score": method_scores["hybrid"],
                "rule_rerank_score": method_scores["rerank"],
                "gold_papers": item["gold_papers"],
                "selection_categories": candidate["categories"],
                "score_deltas": {
                    "hybrid_minus_vector": method_scores["hybrid"]
                    - method_scores["vector"],
                    "rerank_minus_hybrid": method_scores["rerank"]
                    - method_scores["hybrid"],
                },
                "top_evidence": evidence,
                "manual_failure_label": "",
                "manual_notes": "",
            }
        )
    return records


def markdown_table(breakdown: dict) -> str:
    columns = ("Fact", "Relation", "Comparison", "Multi-hop", "overall")
    lines = [
        "| Method | Fact | Relation | Comparison | Multi-hop | Overall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    labels = {"vector": "Structured Vector", "hybrid": "Vanilla Hybrid", "rerank": "Hybrid + Rule Rerank"}
    for method, label in labels.items():
        values = [breakdown[column][method] for column in columns]
        lines.append(f"| {label} | " + " | ".join(f"{value:.4f}" for value in values) + " |")
    return "\n".join(lines)


def review_markdown(records: list[dict], threshold: float) -> str:
    lines = [
        "# Core32 Failure Review Candidates",
        "",
        "This file is a deterministic review queue, not an automated failure analysis.",
        f"Selection threshold: absolute judge-score delta >= {threshold:.2f}.",
        "Manual labels and notes are intentionally blank in the JSONL artifact.",
        "",
    ]
    for record in records:
        lines.extend(
            [
                f"## {record['qid']} — {record['qtype']}",
                "",
                f"Categories: {', '.join(record['selection_categories'])}",
                "",
                f"Question: {record['question']}",
                "",
                f"Gold papers: {', '.join(record['gold_papers'])}",
                "",
                f"Scores: Vector={record['vector_score']:.2f}, Hybrid={record['hybrid_score']:.2f}, Rule Rerank={record['rule_rerank_score']:.2f}",
                "",
            ]
        )
        for method, evidence in record["top_evidence"].items():
            lines.extend([f"### {method}", ""])
            for rank, item in enumerate(evidence, start=1):
                lines.extend(
                    [
                        f"{rank}. `{item['paper_id']}` / `{item['source_type']}` / `{item['section']}`",
                        "",
                        f"   {item['content_preview']}",
                        "",
                    ]
                )
        lines.extend(["Manual failure label:", "", "Manual notes:", ""])
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default="benchmark/gnn_qa_core32_generation_stratified.jsonl"
    )
    parser.add_argument("--eval", default="results/final/generation_eval.csv")
    parser.add_argument("--outputs", default="results/final/generation_outputs.jsonl")
    parser.add_argument(
        "--breakdown-output", default="results/experiments/core32_qtype_breakdown.json"
    )
    parser.add_argument(
        "--review-jsonl", default="results/experiments/core32_failure_review.jsonl"
    )
    parser.add_argument(
        "--review-md", default="results/experiments/core32_failure_review.md"
    )
    parser.add_argument(
        "--expect",
        action="append",
        default=[],
        metavar="METHOD=SCORE",
        help="Check a displayed four-decimal canonical overall before writing outputs.",
    )
    parser.add_argument("--failure-threshold", type=float, default=0.20)
    args = parser.parse_args()

    benchmark_path = Path(args.benchmark)
    eval_path = Path(args.eval)
    outputs_path = Path(args.outputs)
    benchmark = read_jsonl(benchmark_path)
    if len(benchmark) != 32:
        raise ValueError(f"Expected Core32 benchmark, found {len(benchmark)} queries")
    scores = load_scores(eval_path, benchmark)
    breakdown = qtype_breakdown(benchmark, scores)
    expected = parse_expected(args.expect)
    validate_expected(breakdown["overall"], expected)

    output_rows = read_jsonl(outputs_path)
    outputs = {(row["qid"], row["method"]): row for row in output_rows}
    candidates = select_failure_candidates(
        benchmark, scores, args.failure_threshold
    )
    review_records = build_review_records(candidates, scores, outputs)

    breakdown_output = Path(args.breakdown_output)
    review_jsonl = Path(args.review_jsonl)
    review_md = Path(args.review_md)
    for path in (breakdown_output, review_jsonl, review_md):
        path.parent.mkdir(parents=True, exist_ok=True)
    breakdown_output.write_text(
        json.dumps(breakdown, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with review_jsonl.open("w", encoding="utf-8") as handle:
        for record in review_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    review_md.write_text(
        review_markdown(review_records, args.failure_threshold), encoding="utf-8"
    )

    print(markdown_table(breakdown))
    print(f"\nFailure-review candidates: {len(review_records)}")
    print(f"Breakdown: {breakdown_output}")
    print(f"Review JSONL: {review_jsonl}")
    print(f"Review Markdown: {review_md}")


if __name__ == "__main__":
    main()
