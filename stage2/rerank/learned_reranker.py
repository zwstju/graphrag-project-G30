from __future__ import annotations

import json
from dataclasses import is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

from stage2.rerank.feature_extractor import (
    FEATURE_NAMES,
    EvidenceFeatureExtractor,
)


def _set_rerank_score(evidence: Any, score: float) -> Any:
    score = float(score)

    if isinstance(evidence, dict):
        out = dict(evidence)
        out["rerank_score"] = score
        return out

    if is_dataclass(evidence):
        try:
            return replace(evidence, rerank_score=score)
        except TypeError:
            pass

    try:
        setattr(evidence, "rerank_score", score)
        return evidence
    except Exception:
        return evidence


class LearnedEvidenceReranker:
    """
    Pairwise-learning-to-rank reranker.

    score(e, q) = w^T phi(e, q)

    where phi contains:
      semantic, entity, relation, section, source

    The five weights are loaded from a trained JSON model instead of being
    manually specified.
    """

    def __init__(
        self,
        embedding_model: Any,
        model_path: str | Path,
    ):
        self.extractor = EvidenceFeatureExtractor(embedding_model)

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Learned reranker model not found: {model_path}. "
                "Run scripts/train_learned_reranker.py first."
            )

        payload = json.loads(model_path.read_text(encoding="utf-8"))
        names = payload.get("feature_names")
        if names != FEATURE_NAMES:
            raise ValueError(
                f"Feature mismatch. model={names}, code={FEATURE_NAMES}"
            )

        self.weights = np.asarray(payload["weights"], dtype=np.float32)
        if self.weights.shape != (len(FEATURE_NAMES),):
            raise ValueError(
                f"Expected {len(FEATURE_NAMES)} weights, got {self.weights.shape}"
            )

        self.model_info = payload

    def score(self, query: str, evidences: Sequence[Any]) -> np.ndarray:
        x = self.extractor.transform(query, evidences)
        if len(x) == 0:
            return np.zeros((0,), dtype=np.float32)
        return x @ self.weights

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
