from __future__ import annotations

import math
from dataclasses import is_dataclass, replace
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def _plain(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        try:
            from dataclasses import asdict

            return asdict(value)
        except Exception:
            pass
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {"content": str(value)}


def evidence_dict(e: Any) -> Dict[str, Any]:
    d = _plain(e)
    md = d.get("metadata")
    if not isinstance(md, dict):
        md = {}

    content = (
        d.get("content")
        or d.get("text")
        or md.get("content")
        or md.get("text")
        or ""
    )
    paper_id = d.get("paper_id") or md.get("paper_id") or ""
    paper_ids = d.get("paper_ids") or md.get("paper_ids") or []
    if isinstance(paper_ids, str):
        paper_ids = [paper_ids]
    if paper_id and paper_id not in paper_ids:
        paper_ids = [paper_id] + paper_ids

    return {
        **d,
        "metadata": md,
        "content": str(content),
        "paper_id": str(paper_id),
        "paper_ids": [str(x) for x in paper_ids],
        "title": str(d.get("title") or md.get("title") or ""),
        "section": str(d.get("section") or md.get("section") or "Unknown"),
        "source_type": str(
            d.get("source_type") or md.get("source_type") or "vector"
        ).lower(),
        "retrieval_score": d.get(
            "retrieval_score", md.get("retrieval_score", d.get("score", None))
        ),
    }


def _set_rerank_score(evidence: Any, score: float) -> Any:
    score = float(score)
    if isinstance(evidence, dict):
        out = dict(evidence)
        out["rerank_score"] = score
        out["cross_encoder_score"] = score
        return out
    if is_dataclass(evidence):
        try:
            return replace(evidence, rerank_score=score)
        except Exception:
            pass
    for name in ("rerank_score", "cross_encoder_score"):
        try:
            setattr(evidence, name, score)
        except Exception:
            pass
    return evidence


def format_passage(evidence: Any) -> str:
    """
    Give the pretrained reranker enough context to compare heterogeneous
    evidence without changing the underlying evidence itself.
    """
    d = evidence_dict(evidence)
    source = d["source_type"].upper()
    paper = d["paper_id"] or "unknown"
    section = d["section"] or "Unknown"
    return (
        f"Evidence type: {source}\n"
        f"Paper: {paper}\n"
        f"Section: {section}\n"
        f"Content: {d['content']}"
    )


class CrossEncoderEvidenceReranker:
    """
    Pretrained multilingual cross-encoder evidence reranker.

    Default model:
        BAAI/bge-reranker-v2-m3

    It jointly encodes (query, evidence) and directly predicts relevance.
    No project-specific training is required.

    CPU is the safe default because Stage2 already keeps BGE-M3 retrieval
    embeddings/model on GPU and the user's RTX 4060 has ~8 GB VRAM.
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cpu",
        batch_size: int = 4,
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.model_name = model_name
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.max_length = int(max_length)
        self.normalize = bool(normalize)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-x))

    def score(self, query: str, evidences: Sequence[Any]) -> np.ndarray:
        if not evidences:
            return np.zeros((0,), dtype=np.float32)

        passages = [format_passage(e) for e in evidences]
        raw_scores = []
        for start in range(0, len(passages), self.batch_size):
            batch = passages[start : start + self.batch_size]
            pairs = [[query, passage] for passage in batch]
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.inference_mode():
                logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
            raw_scores.extend(logits.detach().cpu().tolist())

        scores = np.asarray(raw_scores, dtype=np.float32)
        if self.normalize:
            scores = self._sigmoid(scores).astype(np.float32)
        return scores

    def rerank(
        self,
        query: str,
        evidences: Sequence[Any],
        top_k: int,
    ) -> List[Any]:
        if not evidences:
            return []

        scores = self.score(query, evidences)
        order = np.argsort(-scores)
        out = []
        for idx in order[:top_k]:
            out.append(
                _set_rerank_score(evidences[int(idx)], float(scores[int(idx)]))
            )
        return out
