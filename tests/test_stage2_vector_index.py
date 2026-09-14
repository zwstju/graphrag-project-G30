import json

import faiss
import numpy as np
import pytest

from stage2.retrieval.vector import VectorIndex, resolve_device


class FakeEmbeddingModel:
    def encode(
        self,
        texts,
        *,
        batch_size,
        normalize_embeddings,
        show_progress_bar,
    ):
        assert batch_size > 0
        assert normalize_embeddings is True
        return np.asarray(
            [[1.0, float(index % 2)] for index, _ in enumerate(texts)],
            dtype="float32",
        ) / np.asarray(
            [np.linalg.norm([1.0, float(index % 2)]) for index, _ in enumerate(texts)],
            dtype="float32",
        )[:, None]


def vector_index_without_loading_model() -> VectorIndex:
    vector_index = VectorIndex.__new__(VectorIndex)
    vector_index.model_name = "test/model"
    vector_index.requested_device = "cpu"
    vector_index.device = "cpu"
    vector_index.model = FakeEmbeddingModel()
    vector_index.index = None
    vector_index.meta = []
    vector_index.manifest = None
    return vector_index


def write_chunks(path) -> None:
    chunks = [
        {"paper_id": "p1", "chunk_id": "c1", "section": "Method", "text": "one"},
        {"paper_id": "p2", "chunk_id": "c2", "section": "Results", "text": "two"},
    ]
    path.write_text(
        "".join(json.dumps(chunk) + "\n" for chunk in chunks),
        encoding="utf-8",
    )


def test_resolve_device_never_silently_falls_back(monkeypatch):
    monkeypatch.setattr("stage2.retrieval.vector.torch.cuda.is_available", lambda: False)
    assert resolve_device("auto") == "cpu"
    assert resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError, match="No CPU fallback"):
        resolve_device("cuda")


def test_build_and_load_validate_manifest(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.index"
    metadata_path = tmp_path / "vector_meta.jsonl"
    manifest_path = tmp_path / "vector_manifest.json"
    write_chunks(corpus)

    built = vector_index_without_loading_model()
    assert built.build(
        str(corpus),
        str(index_path),
        str(metadata_path),
        str(manifest_path),
        batch_size=1,
    ) == 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["chunk_count"] == 2
    assert manifest["embedding_dimension"] == 2
    assert manifest["normalized"] is True
    assert manifest["faiss_index_type"] == "IndexFlatIP"

    loaded = vector_index_without_loading_model().load(
        str(index_path),
        str(metadata_path),
        str(manifest_path),
        str(corpus),
    )
    assert loaded.index.ntotal == len(loaded.meta) == 2


def test_load_rejects_index_metadata_count_mismatch(tmp_path):
    index_path = tmp_path / "vector.index"
    metadata_path = tmp_path / "vector_meta.jsonl"
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0]], dtype="float32"))
    faiss.write_index(index, str(index_path))
    metadata_path.write_text(
        json.dumps({"text": "one"}) + "\n" + json.dumps({"text": "two"}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="index/metadata mismatch"):
        vector_index_without_loading_model().load(str(index_path), str(metadata_path))


def test_load_rejects_changed_corpus(tmp_path):
    corpus = tmp_path / "chunks.jsonl"
    index_path = tmp_path / "vector.index"
    metadata_path = tmp_path / "vector_meta.jsonl"
    manifest_path = tmp_path / "vector_manifest.json"
    write_chunks(corpus)
    vector_index_without_loading_model().build(
        str(corpus),
        str(index_path),
        str(metadata_path),
        str(manifest_path),
        batch_size=1,
    )
    corpus.write_text(corpus.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="corpus SHA-256"):
        vector_index_without_loading_model().load(
            str(index_path),
            str(metadata_path),
            str(manifest_path),
            str(corpus),
        )
