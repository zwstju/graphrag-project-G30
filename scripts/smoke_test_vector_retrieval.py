import argparse
import json
import os
import re

import torch
from dotenv import load_dotenv

from stage2.retrieval.vector import VectorIndex, gpu_name, resolve_device


DEFAULT_QUERIES = [
    "VGAE 和 GCN 有什么关系？",
    "Which model uses GCN as an encoder?",
    "Which datasets are used to evaluate GraphSAGE?",
    "GCN 和 GAT 在邻居权重分配上有什么区别？",
    "哪个模型既使用 GCN 编码器，又通过重构邻接关系用于 link prediction？",
]


def preview(text: str, limit: int) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Qualitative sanity check for the persisted Stage 2 vector index."
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--preview-chars", type=int, default=240)
    parser.add_argument(
        "--query",
        action="append",
        help="Custom query; repeat this option to test multiple queries.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    model_name = os.getenv("EMBED_MODEL", config["embedding_model"])
    requested_device = args.device or config.get("embedding_device", "auto")
    print("Qualitative smoke test only; these results are not evaluation metrics.")
    print(f"Requested device: {requested_device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    try:
        resolved_device = resolve_device(requested_device)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc
    print(f"Resolved device: {resolved_device}")
    if resolved_device == "cuda":
        print(f"GPU name: {gpu_name(resolved_device)}")

    vector_index = VectorIndex(model_name, device=resolved_device).load(
        config["faiss_path"],
        config["vector_meta_path"],
        config["vector_manifest_path"],
        config["retrieval_chunks_path"],
    )
    print(
        "Index integrity: "
        f"{vector_index.index.ntotal} vectors == {len(vector_index.meta)} metadata rows"
    )

    queries = args.query or DEFAULT_QUERIES
    for query_number, query in enumerate(queries, start=1):
        print(f"\nQuery {query_number}: {query}")
        for rank, evidence in enumerate(vector_index.search(query, args.top_k), start=1):
            chunk_id = evidence.metadata.get("chunk_id", "")
            print(
                f"  rank={rank} score={evidence.retrieval_score:.6f} "
                f"paper_id={evidence.paper_id} section={evidence.section}"
            )
            print(f"  title={evidence.title}")
            print(f"  chunk_id={chunk_id}")
            print(f"  preview={preview(evidence.content, args.preview_chars)}")


if __name__ == "__main__":
    main()
