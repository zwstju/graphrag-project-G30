import argparse
import json
import os

import torch
from dotenv import load_dotenv

from stage2.retrieval.vector import VectorIndex, gpu_name, resolve_device


def inspect_chunks(path: str) -> tuple[int, int]:
    """统计检索语料总块数及其中的 References 块数。"""
    chunk_count = 0
    references_count = 0
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            chunk_count += 1
            if chunk.get("section") == "References":
                references_count += 1
    return chunk_count, references_count


def main() -> None:
    """按配置构建持久化 BGE-M3/FAISS 索引并执行完整性检查。"""
    load_dotenv()
    parser = argparse.ArgumentParser(description="Build the persisted Stage 2 BGE-M3 index.")
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--batch-size", type=int)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = json.load(handle)

    model_name = os.getenv("EMBED_MODEL", config["embedding_model"])
    requested_device = args.device or config.get("embedding_device", "auto")
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else config.get("embedding_batch_size", 16)
    )
    if batch_size <= 0:
        raise SystemExit("ERROR: --batch-size must be greater than zero")

    chunk_count, references_count = inspect_chunks(config["retrieval_chunks_path"])
    if references_count:
        raise SystemExit(
            "ERROR: Retrieval corpus contains "
            f"{references_count} canonical References chunks; rebuild it before indexing."
        )
    cuda_available = torch.cuda.is_available()

    print(f"Requested device: {requested_device}")
    print(f"CUDA available: {cuda_available}")
    try:
        resolved_device = resolve_device(requested_device)
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    print(f"Resolved device: {resolved_device}")
    if resolved_device == "cuda":
        print(f"GPU name: {gpu_name(resolved_device)}")
    print(f"Embedding model: {model_name}")
    print(f"Chunk count: {chunk_count}")
    print(f"References chunks: {references_count}")
    print(f"Embedding batch size: {batch_size}")

    vector_index = VectorIndex(model_name, device=resolved_device)
    chunk_count = vector_index.build(
        config["retrieval_chunks_path"],
        config["faiss_path"],
        config["vector_meta_path"],
        config["vector_manifest_path"],
        batch_size=batch_size,
    )
    vector_index.load(
        config["faiss_path"],
        config["vector_meta_path"],
        config["vector_manifest_path"],
        config["retrieval_chunks_path"],
    )

    print(f"Indexed chunks: {chunk_count}")
    print(
        "Persisted integrity check: "
        f"index.ntotal={vector_index.index.ntotal}, "
        f"metadata_count={len(vector_index.meta)}"
    )
    print(f"Embedding dimension: {vector_index.index.d}")
    print("Normalized embeddings: true")
    print(f"FAISS index type: {type(vector_index.index).__name__}")
    print(f"FAISS index path: {config['faiss_path']}")
    print(f"Metadata path: {config['vector_meta_path']}")
    print(f"Build manifest path: {config['vector_manifest_path']}")


if __name__ == "__main__":
    main()
