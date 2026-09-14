import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from stage2.schema import Evidence


SUPPORTED_DEVICES = {"auto", "cuda", "cpu"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_device(requested_device: str) -> str:
    requested = requested_device.casefold()
    if requested not in SUPPORTED_DEVICES:
        choices = ", ".join(sorted(SUPPORTED_DEVICES))
        raise ValueError(f"Unsupported device '{requested_device}'. Choose one of: {choices}")

    cuda_available = torch.cuda.is_available()
    if requested == "cuda" and not cuda_available:
        raise RuntimeError(
            "CUDA was explicitly requested, but torch.cuda.is_available() is False "
            "in this execution environment. No CPU fallback was performed."
        )
    if requested == "auto":
        return "cuda" if cuda_available else "cpu"
    return requested


def gpu_name(device: str) -> str | None:
    if device != "cuda":
        return None
    return torch.cuda.get_device_name(torch.cuda.current_device())


def _temporary_path(target: Path) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    handle.close()
    return Path(handle.name)


class VectorIndex:
    def __init__(self, model_name: str = "BAAI/bge-m3", device: str = "auto"):
        self.model_name = model_name
        self.requested_device = device
        self.device = resolve_device(device)
        self.model = SentenceTransformer(model_name, device=self.device)
        self.index = None
        self.meta = []
        self.manifest = None

    def _encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        vecs = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )
        return np.asarray(vecs, dtype="float32")

    def build(
        self,
        chunks_path: str,
        faiss_path: str,
        meta_path: str,
        manifest_path: str,
        batch_size: int = 16,
    ) -> int:
        chunks = []
        with open(chunks_path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at {chunks_path}:{line_number}: {exc}"
                    ) from exc
                if not chunk.get("text"):
                    raise ValueError(f"Missing chunk text at {chunks_path}:{line_number}")
                chunks.append(chunk)

        if not chunks:
            raise ValueError(f"No chunks found in {chunks_path}")
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        texts = [chunk["text"] for chunk in chunks]
        matrix = self._encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
            raise RuntimeError(
                f"Unexpected embedding shape {matrix.shape} for {len(chunks)} chunks"
            )
        if not np.isfinite(matrix).all():
            raise RuntimeError("Embedding matrix contains non-finite values")

        norms = np.linalg.norm(matrix, axis=1)
        if not np.allclose(norms, 1.0, atol=1e-3):
            raise RuntimeError("Document embeddings are not L2-normalized")

        index = faiss.IndexFlatIP(matrix.shape[1])
        index.add(matrix)

        index_path = Path(faiss_path)
        metadata_path = Path(meta_path)
        build_manifest_path = Path(manifest_path)
        index_temp = _temporary_path(index_path)
        metadata_temp = _temporary_path(metadata_path)
        manifest_temp = _temporary_path(build_manifest_path)

        try:
            faiss.write_index(index, str(index_temp))
            with metadata_temp.open("w", encoding="utf-8") as handle:
                for chunk in chunks:
                    handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            manifest = {
                "manifest_version": 1,
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "embedding_model": self.model_name,
                "build_device": self.device,
                "embedding_batch_size": batch_size,
                "corpus_path": str(Path(chunks_path)),
                "corpus_sha256": sha256_file(chunks_path),
                "chunk_count": len(chunks),
                "embedding_dimension": int(matrix.shape[1]),
                "normalized": True,
                "faiss_index_type": type(index).__name__,
                "index_path": str(index_path),
                "index_sha256": sha256_file(index_temp),
                "metadata_path": str(metadata_path),
                "metadata_sha256": sha256_file(metadata_temp),
            }
            with manifest_temp.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, ensure_ascii=False, indent=2)
                handle.write("\n")

            os.replace(index_temp, index_path)
            os.replace(metadata_temp, metadata_path)
            os.replace(manifest_temp, build_manifest_path)
        finally:
            for temp_path in (index_temp, metadata_temp, manifest_temp):
                if temp_path.exists():
                    temp_path.unlink()

        self.index = index
        self.meta = chunks
        self.manifest = manifest
        return len(chunks)

    def load(
        self,
        faiss_path: str,
        meta_path: str,
        manifest_path: str | None = None,
        corpus_path: str | None = None,
    ):
        index_path = Path(faiss_path)
        metadata_path = Path(meta_path)
        index = faiss.read_index(str(index_path))
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = [json.loads(line) for line in handle if line.strip()]

        if index.ntotal != len(metadata):
            raise RuntimeError(
                "Vector index/metadata mismatch: "
                f"index.ntotal={index.ntotal}, metadata_count={len(metadata)}"
            )

        manifest = None
        if manifest_path is not None:
            build_manifest_path = Path(manifest_path)
            if not build_manifest_path.exists():
                raise FileNotFoundError(
                    f"Vector build manifest not found: {build_manifest_path}"
                )
            with build_manifest_path.open(encoding="utf-8") as handle:
                manifest = json.load(handle)
            self._validate_manifest(
                manifest,
                index=index,
                metadata=metadata,
                index_path=index_path,
                metadata_path=metadata_path,
                corpus_path=Path(corpus_path) if corpus_path else None,
            )

        self.index = index
        self.meta = metadata
        self.manifest = manifest
        return self

    def _validate_manifest(
        self,
        manifest: dict,
        *,
        index,
        metadata: list,
        index_path: Path,
        metadata_path: Path,
        corpus_path: Path | None,
    ) -> None:
        required = {
            "embedding_model",
            "corpus_sha256",
            "chunk_count",
            "embedding_dimension",
            "normalized",
            "faiss_index_type",
            "index_sha256",
            "metadata_sha256",
        }
        missing = sorted(required - manifest.keys())
        if missing:
            raise RuntimeError(f"Vector manifest is missing required fields: {missing}")
        if manifest["embedding_model"] != self.model_name:
            raise RuntimeError(
                "Vector model mismatch: "
                f"manifest={manifest['embedding_model']!r}, runtime={self.model_name!r}"
            )
        if manifest["chunk_count"] != len(metadata):
            raise RuntimeError(
                "Manifest/metadata count mismatch: "
                f"manifest={manifest['chunk_count']}, metadata={len(metadata)}"
            )
        if manifest["embedding_dimension"] != index.d:
            raise RuntimeError(
                "Manifest/index dimension mismatch: "
                f"manifest={manifest['embedding_dimension']}, index={index.d}"
            )
        if manifest["normalized"] is not True:
            raise RuntimeError("Vector manifest does not declare normalized embeddings")
        if manifest["faiss_index_type"] != type(index).__name__:
            raise RuntimeError(
                "Manifest/index type mismatch: "
                f"manifest={manifest['faiss_index_type']}, runtime={type(index).__name__}"
            )
        if manifest["index_sha256"] != sha256_file(index_path):
            raise RuntimeError("FAISS index SHA-256 does not match the build manifest")
        if manifest["metadata_sha256"] != sha256_file(metadata_path):
            raise RuntimeError("Vector metadata SHA-256 does not match the build manifest")
        if corpus_path is not None:
            if not corpus_path.exists():
                raise FileNotFoundError(f"Retrieval corpus not found: {corpus_path}")
            if manifest["corpus_sha256"] != sha256_file(corpus_path):
                raise RuntimeError(
                    "Retrieval corpus SHA-256 does not match the vector build manifest"
                )

    def search(self, query: str, top_k: int = 12) -> List[Evidence]:
        if self.index is None:
            raise RuntimeError("VectorIndex not loaded/built")
        query_matrix = self._encode([query], batch_size=1)
        scores, ids = self.index.search(query_matrix, top_k)
        output = []
        for rank, (score, index_id) in enumerate(zip(scores[0], ids[0])):
            if index_id < 0:
                continue
            metadata = self.meta[index_id]
            output.append(
                Evidence(
                    evidence_id=f"V{rank + 1}",
                    source_type="vector",
                    content=metadata["text"],
                    paper_id=metadata.get("paper_id", ""),
                    title=metadata.get("title", ""),
                    section=metadata.get("section", ""),
                    retrieval_score=float(score),
                    metadata={"chunk_id": metadata.get("chunk_id", "")},
                )
            )
        return output
