from __future__ import annotations

import argparse
import json
from pathlib import Path

from stage2.pipeline import Stage2System
from stage2.rerank.learned_reranker import LearnedEvidenceReranker
from stage2.rerank.feature_extractor import evidence_dict


QUESTIONS = [
    "Which model uses GCN as an encoder?",
    "Which datasets are used to evaluate GraphSAGE?",
    "GCN 和 GAT 在邻居权重分配上有什么区别？",
    "GCN 和 GraphSAGE 在归纳能力上有什么区别？",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.json")
    ap.add_argument("--model", default="models/evidence_reranker_ltr.json")
    ap.add_argument("--candidate-k", type=int, default=12)
    ap.add_argument("--top-k", type=int, default=6)
    args = ap.parse_args()

    system = Stage2System(args.config)
    learned = LearnedEvidenceReranker(
        system.vector.model,
        args.model,
    )

    for q in QUESTIONS:
        v = system.vector.search(q, args.candidate_k)
        g = system.graph.search(q, args.candidate_k)
        candidates = list(v) + list(g)

        ranked = learned.rerank(q, candidates, args.top_k)

        print("\n" + "=" * 84)
        print(q)
        for i, e in enumerate(ranked, 1):
            d = evidence_dict(e)
            score = getattr(e, "rerank_score", None)
            if isinstance(e, dict):
                score = e.get("rerank_score")
            print(
                f"{i:02d} "
                f"score={float(score or 0):.4f} "
                f"type={d['source_type']:<7s} "
                f"paper={d['paper_id']:<18s} "
                f"section={d['section']}"
            )
            print("   ", d["content"][:180].replace("\n", " "))


if __name__ == "__main__":
    main()
