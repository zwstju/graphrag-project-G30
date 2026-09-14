"""Build a deterministic Core69 generation subset with eight items per qtype."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from stage2.retrieval.vector import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core69_audit_filtered.jsonl")
    parser.add_argument("--output", default="benchmark/gnn_qa_core32_generation_stratified.jsonl")
    parser.add_argument("--manifest", default="results/final/core32_generation_manifest.json")
    parser.add_argument("--per-class", type=int, default=8)
    args = parser.parse_args()
    source = Path(args.benchmark)
    items = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups = defaultdict(list)
    for position, item in enumerate(items):
        groups[item["qtype"]].append((position, item))
    selected_positions = set()
    for label in ("fact", "relation", "comparison", "multi_hop"):
        ranked = sorted(
            groups[label],
            key=lambda pair: (hashlib.sha256(pair[1]["qid"].encode()).hexdigest(), pair[1]["qid"]),
        )
        if len(ranked) < args.per_class:
            raise RuntimeError(f"Not enough {label} items: {len(ranked)}")
        selected_positions.update(position for position, _ in ranked[:args.per_class])
    selected = [item for position, item in enumerate(items) if position in selected_positions]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    manifest = {
        "name": "Core32 deterministic stratified generation subset",
        "selection": "lowest SHA-256(qid) values within each qtype; source order retained in output",
        "source_benchmark": str(source), "source_sha256": sha256_file(source),
        "output": str(output), "output_sha256": sha256_file(output),
        "total": len(selected), "per_qtype": args.per_class,
        "qids": [item["qid"] for item in selected],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
