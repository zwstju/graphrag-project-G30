import argparse
import json
from collections import Counter
from pathlib import Path


SMOKE_QIDS = [
    "F02",
    "CF19",
    "R01",
    "CR17",
    "C02",
    "C11",
    "M01",
    "M03",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the fixed balanced 8-item Core80 audit smoke subset."
    )
    parser.add_argument("--source", default="benchmark/gnn_qa_core80.jsonl")
    parser.add_argument(
        "--output", default="benchmark/gnn_qa_core80_audit_smoke8.jsonl"
    )
    args = parser.parse_args()

    with open(args.source, encoding="utf-8") as handle:
        by_qid = {
            item["qid"]: item
            for line in handle
            if line.strip()
            for item in [json.loads(line)]
        }

    missing = [qid for qid in SMOKE_QIDS if qid not in by_qid]
    if missing:
        raise RuntimeError(f"Smoke QIDs missing from Core80: {missing}")
    selected = [by_qid[qid] for qid in SMOKE_QIDS]
    type_counts = Counter(item["qtype"] for item in selected)
    expected_types = {"fact", "relation", "comparison", "multi_hop"}
    if set(type_counts) != expected_types or any(count != 2 for count in type_counts.values()):
        raise RuntimeError(f"Smoke subset is not balanced 2-per-type: {type_counts}")
    paper_counts = {len(item["gold_papers"]) for item in selected}
    if not {1, 2, 3}.issubset(paper_counts):
        raise RuntimeError(
            "Smoke subset must contain single-paper, two-paper, and three-paper items"
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Source benchmark: {args.source}")
    print(f"Smoke subset: {output_path}")
    print(f"QIDs: {', '.join(SMOKE_QIDS)}")
    print(f"Question types: {dict(sorted(type_counts.items()))}")
    print(f"Gold-paper cardinalities present: {sorted(paper_counts)}")


if __name__ == "__main__":
    main()
