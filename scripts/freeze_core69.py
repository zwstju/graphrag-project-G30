"""Freeze the deterministic Core69 audit-filtered evaluation set."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument("--audit", default="results/core80_audit.jsonl")
    parser.add_argument("--output", default="benchmark/gnn_qa_core69_audit_filtered.jsonl")
    parser.add_argument("--manifest", default="results/final/core69_manifest.json")
    args = parser.parse_args()

    benchmark_path, audit_path = Path(args.benchmark), Path(args.audit)
    output_path, manifest_path = Path(args.output), Path(args.manifest)
    items = read_jsonl(benchmark_path)
    audits = {record["id"]: record for record in read_jsonl(audit_path)}
    selected = []
    for item in items:
        audit = audits.get(item["qid"])
        if not audit or audit.get("llm_status") != "llm_evidence_supported":
            continue
        frozen = dict(item)
        frozen.update({
            "review_required": True,
            "review_status": "llm_evidence_supported",
            "evaluation_set": "Core69 Audit-Filtered Evaluation Set",
            "inclusion_basis": "LLM evidence audit; not human verification",
            "llm_audit_score": audit.get("llm_support_score"),
            "audit_artifact": str(audit_path),
        })
        if frozen["gold_answer"] != item["gold_answer"]:
            raise RuntimeError(f"gold_answer changed for {item['qid']}")
        selected.append(frozen)

    if len(selected) != 69:
        raise RuntimeError(f"Expected exactly 69 supported items, got {len(selected)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    qtypes = Counter(item["qtype"] for item in selected)
    difficulties = Counter(item["difficulty"] for item in selected)
    papers = sorted({paper for item in selected for paper in item["gold_papers"]})
    manifest = {
        "name": "Core69 Audit-Filtered Evaluation Set",
        "human_verified": False,
        "inclusion_rule": "llm_status == llm_evidence_supported",
        "source_benchmark": str(benchmark_path),
        "source_audit": str(audit_path),
        "output": str(output_path),
        "total": len(selected),
        "qtype_counts": dict(sorted(qtypes.items())),
        "difficulty_counts": dict(sorted(difficulties.items())),
        "gold_paper_count": len(papers),
        "gold_papers": papers,
        "qids": [item["qid"] for item in selected],
        "sha256": sha256(output_path),
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
