"""Create an LLM-evidence-supported review set without claiming human verification.

The legacy filename is retained for compatibility. This script never overwrites a
gold answer and never marks an item manually verified.
"""

import argparse
import json
from pathlib import Path


def read_jsonl(path: str | Path) -> list[dict]:
    """读取候选 benchmark 或 audit JSONL 记录。"""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def create_llm_supported_review_set(
    benchmark_items: list[dict],
    audit_records: list[dict],
    min_score: float,
    audit_path: str,
) -> list[dict]:
    """筛选达到阈值的 LLM-supported 项，同时保持 gold answer 和人工状态不变。"""
    audits = {record["id"]: record for record in audit_records}
    output = []
    for item in benchmark_items:
        audit = audits.get(item["qid"])
        if not audit:
            continue
        if audit.get("llm_status") != "llm_evidence_supported":
            continue
        if not audit.get("llm_supported"):
            continue
        score = float(audit.get("llm_support_score", 0) or 0)
        if score < min_score:
            continue

        candidate_gold_answer = item["gold_answer"]
        promoted = dict(item)
        promoted["review_required"] = True
        promoted["review_status"] = "llm_evidence_supported"
        promoted["llm_audit_score"] = score
        promoted["audit_artifact"] = audit_path
        promoted["suggested_answer"] = audit.get("suggested_answer", "")
        if promoted["gold_answer"] != candidate_gold_answer:
            raise RuntimeError(f"gold_answer changed for {item['qid']}")
        output.append(promoted)
    return output


def main() -> None:
    """解析筛选参数并写出仍需人工复核的 LLM-supported 数据集。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument("--audit", default="results/core80_audit.jsonl")
    parser.add_argument(
        "--output", default="benchmark/gnn_qa_core80_llm_supported.jsonl"
    )
    parser.add_argument("--min-score", type=float, default=0.80)
    args = parser.parse_args()

    if "verified" in Path(args.output).stem.casefold():
        raise SystemExit(
            "ERROR: An LLM-only audit artifact must not use a filename implying verified gold."
        )

    benchmark_items = read_jsonl(args.benchmark)
    audit_records = read_jsonl(args.audit)
    output = create_llm_supported_review_set(
        benchmark_items,
        audit_records,
        args.min_score,
        args.audit,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in output:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(
        f"Selected {len(output)}/{len(benchmark_items)} LLM-evidence-supported "
        f"items with score >= {args.min_score} -> {output_path}"
    )
    print("All selected items remain review_required=true.")
    print("Gold answers modified: 0")
    print("Human review is required before any item may be manually_verified.")


if __name__ == "__main__":
    main()
