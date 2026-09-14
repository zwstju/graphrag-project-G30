"""Build the four required real-query retrieval case studies."""

import argparse
import copy
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv

from scripts.run_retrieval_ablation import direct_select, evidence_key
from stage2.eval.metrics import evidence_precision, retrieval_recall
from stage2.retrieval.graph import GraphRetriever
from stage2.retrieval.vector import VectorIndex
from stage2.rerank.evidence_reranker import EvidenceReranker, qtype


CASES = [
    {
        "case_id": "graphsage_datasets",
        "question": "Which datasets are used to evaluate GraphSAGE?",
        "gold_papers": ["graphsage_2017"],
    },
    {
        "case_id": "gcn_gat_neighbor_weights",
        "question": "GCN 和 GAT 在邻居权重分配上有什么区别？",
        "gold_papers": ["gcn_2017", "gat_2018"],
    },
    {
        "case_id": "gcn_encoder_model",
        "question": "Which model uses GCN as an encoder?",
        "gold_papers": ["vgae_2016"],
    },
    {
        "case_id": "gcn_encoder_adjacency_link_prediction",
        "question": "哪个模型既使用 GCN 编码器，又通过重构邻接关系用于 link prediction？",
        "gold_papers": ["vgae_2016", "gcn_2017"],
    },
]


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structured-config", default="configs/default.json")
    parser.add_argument("--fixed-config", default="configs/fixed_chunk.json")
    parser.add_argument("--generation", default="results/final/generation_outputs.jsonl")
    parser.add_argument("--output", default="results/final/case_studies.jsonl")
    args = parser.parse_args()
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
    generated = {}
    generation_path = Path(args.generation)
    if generation_path.exists():
        for line in generation_path.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record["method"] == "structured_hybrid_rerank" and record.get("answer"):
                generated[record["question"]] = record

    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in CASES:
            query = case["question"]
            print(case["case_id"], flush=True)
            t0=time.perf_counter(); fixed_pool=fixed.search(query, vector_k); fixed_time=time.perf_counter()-t0
            t0=time.perf_counter(); vector_pool=structured.search(query, vector_k); vector_time=time.perf_counter()-t0
            t0=time.perf_counter(); graph_pool=graph.search(query, graph_k); graph_time=time.perf_counter()-t0
            hybrid=direct_select(vector_pool, graph_pool, final_k)
            t0=time.perf_counter(); reranked=reranker.rerank(query, copy.deepcopy(vector_pool+graph_pool), final_k); rerank_time=time.perf_counter()-t0
            methods = {
                "fixed_vector": (fixed_pool[:final_k], fixed_time),
                "structured_vector": (vector_pool[:final_k], vector_time),
                "structured_vanilla_hybrid": (hybrid, vector_time+graph_time),
                "structured_hybrid_rerank": (reranked, vector_time+graph_time+rerank_time),
            }
            method_output={}
            for name,(evidence,latency) in methods.items():
                records=[item.to_dict() for item in evidence]
                method_output[name]={
                    "evidence":records,
                    "evidence_ids":[evidence_key(item) for item in evidence],
                    "paper_retrieval_recall_proxy":retrieval_recall(records,case["gold_papers"]),
                    "paper_evidence_precision_proxy":evidence_precision(records,case["gold_papers"]),
                    "retrieval_latency_s":latency,
                }
            answer_record=generated.get(query)
            observations=[]
            fixed_metrics=method_output["fixed_vector"]
            structured_metrics=method_output["structured_vector"]
            hybrid_metrics=method_output["structured_vanilla_hybrid"]
            rerank_metrics=method_output["structured_hybrid_rerank"]
            if structured_metrics["paper_retrieval_recall_proxy"] > fixed_metrics["paper_retrieval_recall_proxy"]:
                observations.append({
                    "type":"improvement",
                    "comparison":"structured_vector_vs_fixed_vector",
                    "finding":f"Recall proxy improved from {fixed_metrics['paper_retrieval_recall_proxy']:.3f} to {structured_metrics['paper_retrieval_recall_proxy']:.3f}.",
                })
            if rerank_metrics["paper_evidence_precision_proxy"] > hybrid_metrics["paper_evidence_precision_proxy"]:
                observations.append({
                    "type":"improvement",
                    "comparison":"rerank_vs_vanilla_hybrid",
                    "finding":f"Precision proxy improved from {hybrid_metrics['paper_evidence_precision_proxy']:.3f} to {rerank_metrics['paper_evidence_precision_proxy']:.3f}.",
                })
            if rerank_metrics["paper_retrieval_recall_proxy"] < 1.0:
                observations.append({
                    "type":"remaining_failure",
                    "comparison":"reranked_evidence_vs_declared_gold_papers",
                    "finding":f"Reranked evidence covers only {rerank_metrics['paper_retrieval_recall_proxy']:.3f} of declared gold papers.",
                })
            if rerank_metrics["paper_evidence_precision_proxy"] < hybrid_metrics["paper_evidence_precision_proxy"]:
                observations.append({
                    "type":"remaining_failure",
                    "comparison":"rerank_vs_vanilla_hybrid",
                    "finding":f"Reranking reduced precision proxy from {hybrid_metrics['paper_evidence_precision_proxy']:.3f} to {rerank_metrics['paper_evidence_precision_proxy']:.3f}.",
                })
            record=dict(case)
            record.update({
                "inferred_qtype":qtype(query),
                "graph_seed_entities":graph.find_seeds(query),
                "vector_candidate_ids":[evidence_key(item) for item in vector_pool],
                "graph_candidate_ids":[evidence_key(item) for item in graph_pool],
                "methods":method_output,
                "final_answer":answer_record["answer"] if answer_record else "",
                "final_answer_generated":bool(answer_record),
                "final_answer_source":args.generation if answer_record else "not generated in Core32 subset",
                "observations":observations,
            })
            handle.write(json.dumps(record,ensure_ascii=False)+"\n")
    print(f"case studies -> {output}")


if __name__ == "__main__":
    main()
