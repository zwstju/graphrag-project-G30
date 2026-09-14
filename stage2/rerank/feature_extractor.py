from __future__ import annotations

import math
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np


FEATURE_NAMES = [
    "semantic",
    "entity",
    "relation",
    "section",
    "source",
]


MODEL_TERMS = {
    "gcn", "graphsage", "gat", "gin", "vgae", "gae", "deepwalk",
    "node2vec", "mpnn", "diffpool", "sgc", "appnp", "cluster-gcn",
    "cluster_gcn", "graph convolutional network", "graph attention network",
}

DATASET_TERMS = {
    "cora", "citeseer", "pubmed", "reddit", "ppi", "amazon",
    "amazon2m", "nci1", "proteins", "mutag",
}

RELATION_INTENTS = {
    "encoder": {"encoder", "encode", "编码器", "编码"},
    "uses": {"use", "uses", "used", "采用", "使用"},
    "evaluate": {"evaluate", "evaluation", "dataset", "benchmark", "评估", "数据集"},
    "compare": {"compare", "difference", "different", "区别", "比较", "差异"},
    "inductive": {"inductive", "transductive", "unseen", "归纳", "传导", "未见"},
    "link_prediction": {"link prediction", "link-prediction", "链接预测", "边预测"},
    "aggregation": {"aggregate", "aggregation", "aggregator", "邻居", "聚合"},
    "attention": {"attention", "weight", "权重", "注意力"},
    "relation": {"relation", "relationship", "关系"},
    "path": {"path", "multi-hop", "multihop", "路径", "多跳"},
}

SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "background",
    "related work": "related_work",
    "related_work": "related_work",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "model": "method",
    "experiments": "experiments",
    "experiment": "experiments",
    "evaluation": "experiments",
    "results": "results",
    "result": "results",
    "conclusion": "conclusion",
    "appendix": "appendix",
    "unknown": "unknown",
    "references": "references",
}


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return value
    if hasattr(value, "__dict__"):
        return {
            k: v for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return {"content": str(value)}


def evidence_dict(e: Any) -> Dict[str, Any]:
    d = dict(_plain(e))
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
        paper_ids = [paper_id, *paper_ids]

    return {
        **d,
        "metadata": md,
        "content": str(content),
        "paper_id": str(paper_id),
        "paper_ids": [str(x) for x in paper_ids],
        "section": str(d.get("section") or md.get("section") or "Unknown"),
        "source_type": str(
            d.get("source_type")
            or md.get("source_type")
            or "vector"
        ).lower(),
        "retrieval_score": d.get(
            "retrieval_score",
            md.get("retrieval_score", d.get("score", 0.0)),
        ),
    }


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = text.replace("_", " ").replace("/", " ")
    text = re.sub(r"[^0-9a-z\u4e00-\u9fff+\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(text: str) -> set[str]:
    t = _normalize_text(text)
    toks = set(re.findall(r"[a-z][a-z0-9+\-]{1,}|[\u4e00-\u9fff]{2,}", t))
    return toks


def _entity_terms(text: str) -> set[str]:
    norm = _normalize_text(text)
    terms = set()
    for term in MODEL_TERMS | DATASET_TERMS:
        if term in norm:
            terms.add(term)
    # Keep uppercase-like scientific abbreviations after lowercasing via token rules.
    for tok in _tokens(text):
        if tok in MODEL_TERMS or tok in DATASET_TERMS:
            terms.add(tok)
    return terms


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def _infer_question_kind(query: str) -> str:
    q = _normalize_text(query)
    if any(x in q for x in ["compare", "difference", "区别", "比较", "差异"]):
        return "comparison"
    if any(x in q for x in ["dataset", "evaluate", "benchmark", "数据集", "评估"]):
        return "experiment"
    if any(x in q for x in ["why", "how", "method", "architecture", "为什么", "如何", "方法", "结构"]):
        return "method"
    if any(x in q for x in ["relation", "relationship", "uses", "encoder", "关系", "使用", "编码器"]):
        return "relation"
    if any(x in q for x in ["multi-hop", "multihop", "path", "多跳", "路径"]):
        return "multihop"
    return "fact"


def _section_score(query: str, section: str) -> float:
    kind = _infer_question_kind(query)
    sec_norm = _normalize_text(section)
    sec = SECTION_ALIASES.get(sec_norm, sec_norm)

    preferred = {
        "comparison": {"method": 1.0, "experiments": 0.9, "results": 0.85, "introduction": 0.55},
        "experiment": {"experiments": 1.0, "results": 1.0, "method": 0.45, "introduction": 0.35},
        "method": {"method": 1.0, "background": 0.7, "introduction": 0.55, "experiments": 0.45},
        "relation": {"method": 0.9, "background": 0.75, "introduction": 0.6, "experiments": 0.5},
        "multihop": {"method": 0.85, "background": 0.75, "introduction": 0.6, "experiments": 0.5},
        "fact": {"abstract": 0.85, "introduction": 0.7, "method": 0.75, "experiments": 0.65, "results": 0.7},
    }
    if sec == "references":
        return 0.0
    return float(preferred.get(kind, {}).get(sec, 0.35))


def _relation_score(query: str, evidence_text: str, source_type: str) -> float:
    q = _normalize_text(query)
    e = _normalize_text(evidence_text)

    wanted = set()
    for name, terms in RELATION_INTENTS.items():
        if any(term in q for term in terms):
            wanted.add(name)

    if not wanted:
        # Graph evidence still gets a small structural signal.
        return 0.45 if source_type in {"triple", "path"} else 0.25

    hit = 0
    for name in wanted:
        terms = RELATION_INTENTS[name]
        if any(term in e for term in terms):
            hit += 1

    base = hit / max(len(wanted), 1)
    if source_type in {"triple", "path"} and wanted & {"relation", "path", "encoder", "uses", "link_prediction"}:
        base = min(1.0, base + 0.20)
    return float(base)


def _source_match(query: str, source_type: str) -> float:
    kind = _infer_question_kind(query)
    source_type = source_type.lower()

    if kind in {"relation", "multihop"}:
        table = {"vector": 0.45, "triple": 0.90, "path": 1.00}
    elif kind == "comparison":
        table = {"vector": 0.95, "triple": 0.65, "path": 0.55}
    elif kind == "experiment":
        table = {"vector": 1.00, "triple": 0.55, "path": 0.45}
    else:
        table = {"vector": 0.90, "triple": 0.65, "path": 0.55}
    return float(table.get(source_type, 0.50))


class EvidenceFeatureExtractor:
    """
    Five interpretable features for a query-evidence pair:

    semantic: embedding cosine / retrieval similarity
    entity: entity overlap between query and evidence
    relation: relation-intent suitability
    section: section prior conditioned on question kind
    source: source-type suitability conditioned on question kind

    The final weights are NOT hard-coded here; LearnedEvidenceReranker learns them.
    """

    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model

    def _semantic_batch(self, query: str, evidence_rows: Sequence[Dict[str, Any]]) -> List[float]:
        # Prefer actual embedding cosine so Vector / Triple / Path are comparable.
        if self.embedding_model is not None and hasattr(self.embedding_model, "encode"):
            texts = [query] + [r["content"] for r in evidence_rows]
            try:
                emb = self.embedding_model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            except TypeError:
                emb = self.embedding_model.encode(texts)

            emb = np.asarray(emb, dtype=np.float32)
            q = emb[0]
            vals = [_cosine(q, v) for v in emb[1:]]
            # cosine may be [-1, 1], map to [0, 1]
            return [max(0.0, min(1.0, (v + 1.0) / 2.0)) for v in vals]

        # Fallback to retrieval score if encoder is unavailable.
        vals = []
        for row in evidence_rows:
            try:
                s = float(row.get("retrieval_score") or 0.0)
            except Exception:
                s = 0.0
            # Common normalized cosine scores already fall into [0,1].
            vals.append(max(0.0, min(1.0, s)))
        return vals

    def transform(self, query: str, evidences: Sequence[Any]) -> np.ndarray:
        rows = [evidence_dict(e) for e in evidences]
        if not rows:
            return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)

        semantic = self._semantic_batch(query, rows)
        q_entities = _entity_terms(query)

        features = []
        for sem, row in zip(semantic, rows):
            e_entities = _entity_terms(row["content"])
            entity = _jaccard(q_entities, e_entities)

            relation = _relation_score(
                query,
                row["content"],
                row["source_type"],
            )
            section = _section_score(query, row["section"])
            source = _source_match(query, row["source_type"])

            features.append([
                float(sem),
                float(entity),
                float(relation),
                float(section),
                float(source),
            ])

        return np.asarray(features, dtype=np.float32)
