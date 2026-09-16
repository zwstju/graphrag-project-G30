"""Evaluate zero-shot Cross-Encoder reranking on the frozen Core69 set."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from scripts.run_retrieval_ablation import evidence_key
from stage2.eval.metrics import retrieval_recall
from stage2.retrieval.graph import GraphRetriever
from stage2.retrieval.vector import sha256_file
from stage2.rerank.cross_encoder_reranker import CrossEncoderEvidenceReranker
from stage2.schema import Evidence


METHODS = (
    "structured_vector",
    "vanilla_hybrid",
    "rule_rerank",
    "pretrained_cross_encoder_rerank",
)
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


def dedupe_evidence(evidences: list) -> list:
    """Keep the first occurrence of each stable evidence identifier."""
    output = []
    seen = set()
    for evidence in evidences:
        key = evidence_key(evidence)
        if key in seen:
            continue
        seen.add(key)
        output.append(evidence)
    return output


def candidate_pool_hash(evidences: list) -> str:
    payload = json.dumps(
        [evidence_key(evidence) for evidence in evidences],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_vector_metadata(path: Path) -> dict[str, dict]:
    metadata = {}
    for record in read_jsonl(path):
        chunk_id = record.get("chunk_id")
        if not chunk_id or chunk_id in metadata:
            raise ValueError(f"Invalid or duplicate chunk_id in {path}: {chunk_id!r}")
        metadata[chunk_id] = record
    return metadata


def restore_vector_candidates(candidate_ids: list[str], metadata: dict[str, dict]) -> list[Evidence]:
    candidates = []
    for rank, candidate_id in enumerate(candidate_ids, start=1):
        prefix, separator, chunk_id = candidate_id.partition(":")
        if prefix != "vector" or not separator or chunk_id not in metadata:
            raise ValueError(f"Cannot restore frozen vector candidate: {candidate_id}")
        record = metadata[chunk_id]
        candidates.append(
            Evidence(
                evidence_id=f"V{rank}",
                source_type="vector",
                content=record["text"],
                paper_id=record.get("paper_id", ""),
                title=record.get("title", ""),
                section=record.get("section", ""),
                metadata={"chunk_id": chunk_id},
            )
        )
    return candidates


def restore_saved_evidence(records: list[dict]) -> list[Evidence]:
    return [Evidence(**record) for record in records]


def metric_record(evidences: list, gold_papers: list[str], top_k: int) -> dict:
    records = [evidence.to_dict() for evidence in evidences[:top_k]]
    recall = retrieval_recall(records, gold_papers)
    return {
        f"hit_at_{top_k}": float(recall > 0.0),
        f"gold_paper_recall_at_{top_k}": recall,
        "evidence_count": len(records),
        "final_evidence_ids": [evidence_key(evidence) for evidence in evidences[:top_k]],
        "evidence": records,
    }


def aggregate(details: list[dict], top_k: int) -> dict:
    """Aggregate the two required metrics overall and by canonical qtype."""
    groups = {"overall": details}
    for raw_qtype, label in QTYPE_LABELS.items():
        groups[label] = [row for row in details if row["qtype"] == raw_qtype]

    output = {}
    hit_key = f"hit_at_{top_k}"
    recall_key = f"gold_paper_recall_at_{top_k}"
    for group_name, members in groups.items():
        method_stats = {}
        for method in METHODS:
            if not members:
                method_stats[method] = {
                    "n": 0,
                    hit_key: None,
                    recall_key: None,
                }
                continue
            method_stats[method] = {
                "n": len(members),
                hit_key: sum(row["methods"][method][hit_key] for row in members)
                / len(members),
                recall_key: sum(
                    row["methods"][method][recall_key] for row in members
                )
                / len(members),
            }
        output[group_name] = method_stats
    return output


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark", default="benchmark/gnn_qa_core69_audit_filtered.jsonl"
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--retrieval-artifact", default="results/final/retrieval_per_question.jsonl"
    )
    parser.add_argument(
        "--retrieval-manifest", default="results/final/retrieval_manifest.json"
    )
    parser.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate-k", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output-jsonl",
        default="results/experiments/cross_encoder_core69_retrieval.jsonl",
    )
    parser.add_argument(
        "--output-summary",
        default="results/experiments/cross_encoder_core69_retrieval_summary.json",
    )
    args = parser.parse_args()
    if args.candidate_k <= 0 or args.top_k <= 0:
        parser.error("--candidate-k and --top-k must be positive")
    if args.top_k > args.candidate_k * 2:
        parser.error("--top-k exceeds the maximum heterogeneous candidate pool size")
    if args.device != "cpu":
        parser.error("This experiment requires the Cross-Encoder to run on CPU")

    started = datetime.now(timezone.utc)
    benchmark_path = Path(args.benchmark)
    items = read_jsonl(benchmark_path)
    if args.limit is not None:
        if args.limit <= 0:
            parser.error("--limit must be positive")
        items = items[: args.limit]
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    retrieval_artifact_path = Path(args.retrieval_artifact)
    frozen_retrieval = {
        row["qid"]: row for row in read_jsonl(retrieval_artifact_path)
    }
    retrieval_manifest_path = Path(args.retrieval_manifest)
    retrieval_manifest = json.loads(retrieval_manifest_path.read_text(encoding="utf-8"))
    if retrieval_manifest.get("benchmark_sha256") != sha256_file(benchmark_path):
        raise RuntimeError("Frozen retrieval artifact does not match the Core69 benchmark")
    if retrieval_manifest.get("vector_candidates") != args.candidate_k:
        raise RuntimeError("--candidate-k does not match the frozen retrieval artifact")
    if retrieval_manifest.get("final_evidence_k") != args.top_k:
        raise RuntimeError("--top-k does not match the frozen retrieval artifact")
    vector_metadata = load_vector_metadata(Path(config["vector_meta_path"]))
    graph = GraphRetriever(config["triples_path"], config.get("graph_max_hops", 3))
    cross_encoder = CrossEncoderEvidenceReranker(
        model_name=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    details = []
    for index, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{index}/{len(items)}] {item['qid']}: {question}", flush=True)
        source = frozen_retrieval.get(item["qid"])
        if source is None:
            raise ValueError(f"Missing frozen retrieval record for {item['qid']}")
        vector_pool = restore_vector_candidates(
            source["vector_candidate_ids"], vector_metadata
        )
        graph_pool = dedupe_evidence(graph.search(question, args.candidate_k))
        graph_ids = [evidence_key(evidence) for evidence in graph_pool]
        if graph_ids != source["graph_candidate_ids"]:
            raise RuntimeError(
                f"Graph candidate replay mismatch for {item['qid']}"
            )
        candidate_pool = dedupe_evidence(vector_pool + graph_pool)
        cross_started = time.perf_counter()
        cross = cross_encoder.rerank(
            question, copy.deepcopy(candidate_pool), args.top_k
        )
        cross_latency = time.perf_counter() - cross_started
        selected = {
            "structured_vector": restore_saved_evidence(
                source["methods"]["structured_vector"]["evidence"]
            ),
            "vanilla_hybrid": restore_saved_evidence(
                source["methods"]["structured_vanilla_hybrid"]["evidence"]
            ),
            "rule_rerank": restore_saved_evidence(
                source["methods"]["structured_hybrid_rerank"]["evidence"]
            ),
            "pretrained_cross_encoder_rerank": cross,
        }
        if any(len(evidence) != args.top_k for evidence in selected.values()):
            counts = {method: len(evidence) for method, evidence in selected.items()}
            raise RuntimeError(f"Final evidence count mismatch for {item['qid']}: {counts}")

        method_results = {
            method: metric_record(evidence, item["gold_papers"], args.top_k)
            for method, evidence in selected.items()
        }
        method_results["structured_vector"]["latency_s"] = source["methods"][
            "structured_vector"
        ]["retrieval_latency_s"]
        method_results["vanilla_hybrid"]["latency_s"] = source["methods"][
            "structured_vanilla_hybrid"
        ]["retrieval_latency_s"]
        method_results["rule_rerank"]["latency_s"] = source["methods"][
            "structured_hybrid_rerank"
        ]["retrieval_latency_s"]
        method_results["pretrained_cross_encoder_rerank"]["latency_s"] = (
            source["methods"]["structured_vanilla_hybrid"]["retrieval_latency_s"]
            + cross_latency
        )
        details.append(
            {
                "qid": item["qid"],
                "qtype": item["qtype"],
                "question": question,
                "gold_papers": item["gold_papers"],
                "candidate_pool_count": len(candidate_pool),
                "candidate_pool_sha256": candidate_pool_hash(candidate_pool),
                "vector_candidate_ids": [evidence_key(e) for e in vector_pool],
                "graph_candidate_ids": graph_ids,
                "methods": method_results,
            }
        )

    completed = datetime.now(timezone.utc)
    summary = {
        "benchmark_path": str(benchmark_path),
        "benchmark_sha256": sha256_file(benchmark_path),
        "query_count": len(items),
        "candidate_k": args.candidate_k,
        "candidate_pool_max": args.candidate_k * 2,
        "top_k": args.top_k,
        "model_name": args.model,
        "device": str(cross_encoder.device),
        "retrieval_artifact": str(retrieval_artifact_path),
        "retrieval_artifact_sha256": sha256_file(retrieval_artifact_path),
        "retrieval_manifest": str(retrieval_manifest_path),
        "embedding_model": retrieval_manifest["embedding_model"],
        "retrieval_device": retrieval_manifest["embedding_device"],
        "methods": list(METHODS),
        "metric_scope": "paper-level proxy",
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        **aggregate(details, args.top_k),
    }
    output_jsonl = Path(args.output_jsonl)
    output_summary = Path(args.output_summary)
    write_jsonl(output_jsonl, details)
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    output_summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
