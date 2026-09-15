"""Run the controlled four-condition Core69 retrieval ablation."""

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from stage2.eval.metrics import evidence_precision, retrieval_recall
from stage2.retrieval.graph import GraphRetriever
from stage2.retrieval.vector import VectorIndex, sha256_file
from stage2.rerank.evidence_reranker import EvidenceReranker


METHODS = (
    "fixed_vector",
    "structured_vector",
    "structured_vanilla_hybrid",
    "structured_hybrid_rerank",
)


def read_jsonl(path: Path) -> list[dict]:
    """读取指定 JSONL artifact 的全部非空记录。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def evidence_key(evidence) -> str:
    """根据证据类型、内容标识和论文来源生成稳定候选键。"""
    metadata = evidence.metadata or {}
    if evidence.source_type == "vector":
        return f"vector:{metadata.get('chunk_id', evidence.evidence_id)}"
    if evidence.source_type == "triple":
        return "triple:" + "|".join([
            str(metadata.get("source", "")), str(metadata.get("relation", "")),
            str(metadata.get("target", "")), ",".join(evidence.paper_ids),
        ])
    return "path:" + "|".join([
        ">".join(metadata.get("nodes", [])),
        ">".join(metadata.get("relations", [])),
        ",".join(evidence.paper_ids),
    ])


def pool_hash(vector_ids: list[str], graph_ids: list[str]) -> str:
    """计算保持候选顺序的 Vector/Graph 候选池哈希。"""
    payload = json.dumps(
        {"vector_candidate_ids": vector_ids, "graph_candidate_ids": graph_ids},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def direct_select(vector, graph, top_k: int):
    """按固定配额直接拼接两类候选，并在不足时补齐证据预算。"""
    vector_quota = max(1, top_k // 2)
    selected = list(vector[:vector_quota]) + list(graph[:top_k - vector_quota])
    if len(selected) < top_k:
        for evidence in list(vector[vector_quota:]) + list(graph[top_k - vector_quota:]):
            if evidence not in selected:
                selected.append(evidence)
            if len(selected) == top_k:
                break
    return selected[:top_k]


def write_csv(path: Path, rows: list[dict]) -> None:
    """使用首行字段顺序写出 UTF-8 CSV。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], by_qtype: bool) -> list[dict]:
    """按方法及可选题型聚合检索代理指标、证据数和延迟。"""
    groups = defaultdict(list)
    for row in rows:
        key = (row["method"], row["qtype"] if by_qtype else "overall")
        groups[key].append(row)
    output = []
    for (method, qtype_name), members in sorted(groups.items()):
        output.append({
            "method": method,
            "qtype": qtype_name,
            "question_count": len(members),
            "paper_retrieval_recall_proxy": sum(x["paper_retrieval_recall_proxy"] for x in members) / len(members),
            "paper_evidence_precision_proxy": sum(x["paper_evidence_precision_proxy"] for x in members) / len(members),
            "mean_final_evidence_count": sum(x["final_evidence_count"] for x in members) / len(members),
            "mean_retrieval_latency_s": sum(x["retrieval_latency_s"] for x in members) / len(members),
        })
    return output


def main() -> None:
    """运行 Core69 四条件检索消融并保存逐题和聚合 artifact。"""
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="benchmark/gnn_qa_core69_audit_filtered.jsonl")
    parser.add_argument("--structured-config", default="configs/default.json")
    parser.add_argument("--fixed-config", default="configs/fixed_chunk.json")
    parser.add_argument("--output-dir", default="results/final")
    args = parser.parse_args()
    started = datetime.now(timezone.utc)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_path = Path(args.benchmark)
    items = read_jsonl(benchmark_path)
    structured_config = json.loads(Path(args.structured_config).read_text(encoding="utf-8"))
    fixed_config = json.loads(Path(args.fixed_config).read_text(encoding="utf-8"))
    model_name = os.getenv("EMBED_MODEL", structured_config["embedding_model"])
    device = structured_config.get("embedding_device", "auto")

    structured = VectorIndex(model_name, device=device).load(
        structured_config["faiss_path"], structured_config["vector_meta_path"],
        structured_config["vector_manifest_path"], structured_config["retrieval_chunks_path"],
    )
    fixed = VectorIndex(model_name, device=device).load(
        fixed_config["faiss_path"], fixed_config["vector_meta_path"],
        fixed_config["vector_manifest_path"], fixed_config["retrieval_chunks_path"],
    )
    graph = GraphRetriever(structured_config["triples_path"], structured_config.get("graph_max_hops", 3))
    reranker = EvidenceReranker(structured.model, structured_config.get("rerank_weights"))
    vector_k = structured_config.get("vector_candidates", 12)
    graph_k = structured_config.get("graph_candidates", 12)
    final_k = structured_config.get("final_evidence_k", 6)

    rows = []
    details = []
    for index, item in enumerate(items, start=1):
        question = item["question"]
        print(f"[{index}/{len(items)}] {item['qid']}: {question}", flush=True)
        t0 = time.perf_counter()
        fixed_pool = fixed.search(question, vector_k)
        fixed_latency = time.perf_counter() - t0
        t0 = time.perf_counter()
        vector_pool = structured.search(question, vector_k)
        vector_latency = time.perf_counter() - t0
        t0 = time.perf_counter()
        graph_pool = graph.search(question, graph_k)
        graph_latency = time.perf_counter() - t0

        t0 = time.perf_counter()
        hybrid = direct_select(vector_pool, graph_pool, final_k)
        hybrid_select_latency = time.perf_counter() - t0
        t0 = time.perf_counter()
        reranked = reranker.rerank(question, copy.deepcopy(vector_pool + graph_pool), final_k)
        rerank_latency = time.perf_counter() - t0
        methods = {
            "fixed_vector": (fixed_pool[:final_k], fixed_latency),
            "structured_vector": (vector_pool[:final_k], vector_latency),
            "structured_vanilla_hybrid": (hybrid, vector_latency + graph_latency + hybrid_select_latency),
            "structured_hybrid_rerank": (reranked, vector_latency + graph_latency + rerank_latency),
        }
        if any(len(evidence) != final_k for evidence, _ in methods.values()):
            counts = {name: len(value[0]) for name, value in methods.items()}
            raise RuntimeError(f"Final evidence count mismatch for {item['qid']}: {counts}")

        vector_ids = [evidence_key(e) for e in vector_pool]
        graph_ids = [evidence_key(e) for e in graph_pool]
        candidate_hash = pool_hash(vector_ids, graph_ids)
        method_details = {}
        for method, (evidence, latency) in methods.items():
            evidence_dicts = [e.to_dict() for e in evidence]
            row = {
                "qid": item["qid"], "qtype": item["qtype"], "difficulty": item["difficulty"],
                "method": method, "retrieval_metric_scope": "paper-level-proxy",
                "paper_retrieval_recall_proxy": retrieval_recall(evidence_dicts, item["gold_papers"]),
                "paper_evidence_precision_proxy": evidence_precision(evidence_dicts, item["gold_papers"]),
                "final_evidence_count": len(evidence), "retrieval_latency_s": latency,
                "candidate_pool_sha256": candidate_hash if "hybrid" in method else "",
                "vector_candidate_ids": json.dumps(vector_ids, ensure_ascii=False) if "hybrid" in method else "",
                "graph_candidate_ids": json.dumps(graph_ids, ensure_ascii=False) if "hybrid" in method else "",
                "final_evidence_ids": json.dumps([evidence_key(e) for e in evidence], ensure_ascii=False),
            }
            rows.append(row)
            method_details[method] = {
                "retrieval_latency_s": latency,
                "paper_retrieval_recall_proxy": row["paper_retrieval_recall_proxy"],
                "paper_evidence_precision_proxy": row["paper_evidence_precision_proxy"],
                "evidence": evidence_dicts,
                "final_evidence_ids": [evidence_key(e) for e in evidence],
            }
        details.append({
            "qid": item["qid"], "question": question, "qtype": item["qtype"],
            "difficulty": item["difficulty"], "gold_answer": item["gold_answer"],
            "gold_papers": item["gold_papers"], "vector_candidate_ids": vector_ids,
            "graph_candidate_ids": graph_ids, "candidate_pool_sha256": candidate_hash,
            "methods": method_details,
        })

    write_csv(output_dir / "retrieval_ablation.csv", rows)
    write_csv(output_dir / "retrieval_overall.csv", aggregate(rows, False))
    write_csv(output_dir / "retrieval_by_qtype.csv", aggregate(rows, True))
    with (output_dir / "retrieval_per_question.jsonl").open("w", encoding="utf-8") as handle:
        for detail in details:
            handle.write(json.dumps(detail, ensure_ascii=False) + "\n")
    completed = datetime.now(timezone.utc)
    manifest = {
        "benchmark": str(benchmark_path), "benchmark_sha256": sha256_file(benchmark_path),
        "question_count": len(items), "methods": list(METHODS),
        "vector_candidates": vector_k, "graph_candidates": graph_k,
        "final_evidence_k": final_k, "embedding_model": model_name,
        "embedding_device": structured.device,
        "structured_index_manifest": structured_config["vector_manifest_path"],
        "fixed_index_manifest": fixed_config["vector_manifest_path"],
        "started_at_utc": started.isoformat(), "completed_at_utc": completed.isoformat(),
        "runtime_seconds": (completed - started).total_seconds(),
        "candidate_pool_invariant": "Structured Vanilla Hybrid and Structured Hybrid + Rerank share each query's exact Vector and Graph candidate IDs.",
    }
    (output_dir / "retrieval_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest": manifest, "overall": aggregate(rows, False)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
