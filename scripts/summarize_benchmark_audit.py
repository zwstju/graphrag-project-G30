"""Summarize an LLM evidence-audit artifact without changing benchmark labels."""

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from stage2.retrieval.vector import sha256_file


SUCCESS_STATUSES = {"llm_evidence_supported", "llm_evidence_not_supported"}


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def support_stats(records: list[dict]) -> dict:
    successful = [record for record in records if record["llm_status"] in SUCCESS_STATUSES]
    scores = [float(record.get("llm_support_score", 0)) for record in successful]
    supported = sum(
        record["llm_status"] == "llm_evidence_supported" for record in records
    )
    not_supported = sum(
        record["llm_status"] == "llm_evidence_not_supported" for record in records
    )
    failed = sum(record["llm_status"] == "llm_audit_error" for record in records)
    return {
        "total": len(records),
        "successfully_audited": len(successful),
        "llm_evidence_supported": supported,
        "llm_evidence_not_supported": not_supported,
        "audit_failures": failed,
        "candidate_answer_evidence_support_rate": (
            supported / len(successful) if successful else None
        ),
        "mean_support_score_successful_audits": (
            statistics.fmean(scores) if scores else None
        ),
        "median_support_score_successful_audits": (
            statistics.median(scores) if scores else None
        ),
    }


def grouped_stats(records: list[dict], field: str) -> dict:
    groups = defaultdict(list)
    for record in records:
        groups[record[field]].append(record)
    return {key: support_stats(groups[key]) for key in sorted(groups)}


def build_summary(
    records: list[dict],
    benchmark_items: list[dict],
    run_manifest: dict,
    audit_path: Path,
) -> tuple[dict, list[dict]]:
    benchmark_by_id = {item["qid"]: item for item in benchmark_items}
    enriched = []
    for record in records:
        item = benchmark_by_id.get(record["id"])
        if item is None:
            raise RuntimeError(f"Audit QID is absent from benchmark: {record['id']}")
        enriched_record = dict(record)
        enriched_record["difficulty"] = item.get("difficulty", "unknown")
        enriched.append(enriched_record)

    cardinalities = Counter(len(record["gold_papers"]) for record in enriched)
    multi_paper = [record for record in enriched if len(record["gold_papers"]) > 1]
    omitted_gold_papers = []
    quota_violations = []
    for record in multi_paper:
        missing = [
            paper_id
            for paper_id in record["gold_papers"]
            if record.get("selected_evidence_counts", {}).get(paper_id, 0) == 0
        ]
        if missing:
            omitted_gold_papers.append({"id": record["id"], "paper_ids": missing})
        below_quota = [
            paper_id
            for paper_id in record["gold_papers"]
            if record.get("selected_evidence_counts", {}).get(paper_id, 0)
            < record.get("top_k_per_gold_paper", 3)
        ]
        if below_quota:
            quota_violations.append({"id": record["id"], "paper_ids": below_quota})

    selected_evidence = [
        evidence for record in enriched for evidence in record.get("selected_evidence", [])
    ]
    gold_answer_modifications = [
        record["id"]
        for record in enriched
        if record["candidate_gold_answer"]
        != benchmark_by_id[record["id"]]["gold_answer"]
    ]

    total_attempts = sum(int(record.get("audit_attempts", 0)) for record in enriched)
    retries = sum(max(0, int(record.get("audit_attempts", 0)) - 1) for record in enriched)
    empty_responses = sum(
        not response.strip()
        for record in enriched
        for response in record.get("raw_audit_responses", [])
    )
    json_parse_failures = sum(
        "JSONDecodeError" in error
        for record in enriched
        for error in record.get("audit_parse_errors", [])
    )
    transport_failures = sum(
        len(record.get("audit_transport_errors", [])) for record in enriched
    )

    suggested_answer_queue = [
        {
            "id": record["id"],
            "original_candidate_answer": record["candidate_gold_answer"],
            "suggested_answer": record.get("suggested_answer", ""),
            "audit_reason": record.get("reason", ""),
        }
        for record in enriched
        if record["llm_status"] == "llm_evidence_not_supported"
        and record.get("suggested_answer", "").strip()
    ]

    review_queue = []
    for record in enriched:
        reasons = []
        if record["llm_status"] == "llm_evidence_not_supported":
            reasons.append("llm_evidence_not_supported")
        if record["llm_status"] == "llm_audit_error":
            reasons.append("llm_audit_error")
        if (
            record["llm_status"] in SUCCESS_STATUSES
            and float(record.get("llm_support_score", 0)) < 0.8
        ):
            reasons.append("support_score_below_0.8")
        if int(record.get("audit_attempts", 0)) > 1:
            reasons.append("multiple_llm_attempts")
        if reasons:
            queued = dict(record)
            queued["difficulty"] = record["difficulty"]
            queued["manual_review_reasons"] = reasons
            review_queue.append(queued)

    summary = {
        "terminology": {
            "input": "candidate benchmark",
            "automatic_stage": "LLM evidence audit",
            "supported_status": "llm_evidence_supported",
            "next_stage": "manual review",
            "human_approved_status": "manually_verified",
            "metric_name": "candidate-answer evidence support rate",
        },
        "audit_artifact": str(audit_path),
        "audit_artifact_sha256": sha256_file(audit_path),
        "audit_runtime_seconds": run_manifest.get("runtime_seconds"),
        "overall": support_stats(enriched),
        "by_qtype": grouped_stats(enriched, "qtype"),
        "by_difficulty": grouped_stats(enriched, "difficulty"),
        "multi_paper_integrity": {
            "single_paper_questions": cardinalities.get(1, 0),
            "two_paper_questions": cardinalities.get(2, 0),
            "three_paper_questions": cardinalities.get(3, 0),
            "multi_paper_questions": len(multi_paper),
            "gold_paper_omission_violation_count": len(omitted_gold_papers),
            "gold_paper_omission_violations": omitted_gold_papers,
            "per_paper_quota_violation_count": len(quota_violations),
            "per_paper_quota_violations": quota_violations,
        },
        "evidence_integrity": {
            "total_selected_evidence_chunks": len(selected_evidence),
            "references_chunks_selected": sum(
                evidence.get("section") == "References" for evidence in selected_evidence
            ),
            "missing_chunk_ids": sum(
                not evidence.get("chunk_id") for evidence in selected_evidence
            ),
            "missing_paper_ids": sum(
                not evidence.get("paper_id") for evidence in selected_evidence
            ),
            "gold_answer_modification_count": len(gold_answer_modifications),
            "gold_answer_modified_qids": gold_answer_modifications,
            "manual_status_values": sorted(
                {record.get("manual_status", "") for record in enriched}
            ),
        },
        "reliability": {
            "total_llm_api_attempts": total_attempts,
            "retry_count": retries,
            "empty_response_count": empty_responses,
            "json_parse_failure_count": json_parse_failures,
            "transport_failure_count": transport_failures,
            "items_failed_after_all_retries": sum(
                record["llm_status"] == "llm_audit_error" for record in enriched
            ),
            "items_requiring_multiple_attempts": [
                record["id"]
                for record in enriched
                if int(record.get("audit_attempts", 0)) > 1
            ],
        },
        "unsupported_with_suggested_answer_count": len(suggested_answer_queue),
        "unsupported_with_suggested_answers": suggested_answer_queue,
        "unsupported_qids": [
            record["id"]
            for record in enriched
            if record["llm_status"] == "llm_evidence_not_supported"
        ],
        "manual_review_queue_count": len(review_queue),
    }
    return summary, review_queue


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", default="results/core80_audit.jsonl")
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument("--manifest", default="results/core80_audit.manifest.json")
    parser.add_argument("--output", default="results/core80_audit_summary.json")
    parser.add_argument(
        "--review-queue", default="results/core80_manual_review_queue.jsonl"
    )
    args = parser.parse_args()

    audit_path = Path(args.audit)
    records = read_jsonl(audit_path)
    benchmark_items = read_jsonl(args.benchmark)
    with open(args.manifest, encoding="utf-8") as handle:
        run_manifest = json.load(handle)
    summary, review_queue = build_summary(
        records, benchmark_items, run_manifest, audit_path
    )

    output_path = Path(args.output)
    review_path = Path(args.review_queue)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    review_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with review_path.open("w", encoding="utf-8") as handle:
        for record in review_queue:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Summary -> {output_path}")
    print(f"Manual review queue -> {review_path} ({len(review_queue)} items)")


if __name__ == "__main__":
    main()
