from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from stage2.pipeline import Stage2System


ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT / "frontend_stage2"

app = FastAPI(
    title="Evidence-Centric HybridRAG Demo",
    version="2.0.0",
    description="Stage2 live demo API + frontend for the final defense.",
)

# Load once so the vector index / graph are not rebuilt for every request.
SYSTEM = Stage2System(str(ROOT / "configs" / "default.json"))


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    method: Literal["vector", "graph", "hybrid", "rerank", "full"] = "rerank"
    verify: bool = False


def _count_jsonl(path: Path) -> int:
    """统计 JSONL 文件中的非空记录数。"""
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _paper_count() -> int:
    """返回当前标准论文语料包含的论文数量。"""
    path = ROOT / "data" / "processed" / "papers.jsonl"
    return _count_jsonl(path)


def _graph_stats() -> tuple[int, int]:
    """从三元组文件统计去重节点数和边记录数。"""
    path = ROOT / "data" / "graph" / "triples.jsonl"
    if not path.exists():
        return 0, 0

    nodes: set[str] = set()
    edges = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue

            source = (
                row.get("source")
                or row.get("head")
                or row.get("subject")
                or row.get("src")
            )
            target = (
                row.get("target")
                or row.get("tail")
                or row.get("object")
                or row.get("dst")
            )

            if source:
                nodes.add(str(source))
            if target:
                nodes.add(str(target))
            edges += 1

    return len(nodes), edges


def _plain(value: Any) -> Any:
    """递归把数据类、Pydantic 对象和容器转换为可序列化普通值。"""
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict") and callable(value.dict):
        return value.dict()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_plain(v) for v in value]
    if hasattr(value, "__dict__"):
        return {
            str(k): _plain(v)
            for k, v in vars(value).items()
            if not k.startswith("_")
        }
    return value


def _evidence_to_dict(e: Any, rank: int) -> Dict[str, Any]:
    """把不同来源的 Evidence 对象统一转换为前端需要的字典结构。"""
    d = _plain(e)
    if not isinstance(d, dict):
        d = {"content": str(d)}

    metadata = d.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    source_type = (
        d.get("source_type")
        or metadata.get("source_type")
        or "vector"
    )

    content = (
        d.get("content")
        or d.get("text")
        or metadata.get("content")
        or metadata.get("text")
        or ""
    )

    paper_id = (
        d.get("paper_id")
        or metadata.get("paper_id")
        or ""
    )

    paper_ids = d.get("paper_ids") or metadata.get("paper_ids") or []
    if isinstance(paper_ids, str):
        paper_ids = [paper_ids]

    if paper_id and paper_id not in paper_ids:
        paper_ids = [paper_id, *paper_ids]

    return {
        "rank": rank,
        "evidence_id": d.get("evidence_id") or metadata.get("evidence_id") or "",
        "source_type": str(source_type),
        "content": str(content),
        "paper_id": str(paper_id),
        "paper_ids": [str(x) for x in paper_ids],
        "title": d.get("title") or metadata.get("title") or "",
        "section": d.get("section") or metadata.get("section") or "",
        "retrieval_score": d.get("retrieval_score", metadata.get("retrieval_score")),
        "rerank_score": d.get("rerank_score", metadata.get("rerank_score")),
        "metadata": metadata,
    }


TRIPLE_RE = re.compile(
    r"^\s*(.*?)\s+--\s*([A-Za-z0-9_ /:-]+?)\s*-->\s*(.*?)\s*$"
)


def _triple_from_evidence(e: Dict[str, Any]) -> Dict[str, str] | None:
    """从 Evidence 元数据或文本内容中恢复一条图三元组。"""
    md = e.get("metadata") or {}

    source = md.get("source") or md.get("head") or md.get("subject")
    relation = md.get("relation") or md.get("predicate") or md.get("rel")
    target = md.get("target") or md.get("tail") or md.get("object")

    if source and relation and target:
        return {
            "source": str(source),
            "relation": str(relation),
            "target": str(target),
            "paper_id": e.get("paper_id", ""),
        }

    m = TRIPLE_RE.match(e.get("content", ""))
    if m:
        return {
            "source": m.group(1).strip(),
            "relation": m.group(2).strip(),
            "target": m.group(3).strip(),
            "paper_id": e.get("paper_id", ""),
        }

    return None


def _graph_from_evidence(evidence: List[Dict[str, Any]]) -> Dict[str, Any]:
    """将三元组和路径证据转换为去重后的前端图节点与边。"""
    nodes: Dict[str, Dict[str, str]] = {}
    edges: List[Dict[str, str]] = []

    def add_node(node_id: str, node_type: str = "entity") -> None:
        """按节点 ID 将实体或论文节点加入当前可视化图。"""
        if not node_id:
            return
        nodes.setdefault(
            node_id,
            {
                "id": node_id,
                "label": node_id,
                "type": node_type,
            },
        )

    for e in evidence:
        source_type = e.get("source_type", "").lower()
        md = e.get("metadata") or {}

        if source_type == "triple":
            t = _triple_from_evidence(e)
            if t:
                add_node(t["source"], "entity")
                add_node(t["target"], "entity")
                edges.append(t)

        elif source_type == "path":
            path_nodes = (
                md.get("nodes")
                or md.get("path_nodes")
                or []
            )
            relations = (
                md.get("relations")
                or md.get("path_relations")
                or []
            )
            if isinstance(path_nodes, list) and len(path_nodes) >= 2:
                for n in path_nodes:
                    add_node(str(n), "entity")
                for i in range(len(path_nodes) - 1):
                    rel = (
                        relations[i]
                        if i < len(relations)
                        else "RELATED_TO"
                    )
                    edges.append(
                        {
                            "source": str(path_nodes[i]),
                            "relation": str(rel),
                            "target": str(path_nodes[i + 1]),
                            "paper_id": e.get("paper_id", ""),
                        }
                    )

        # Add paper provenance as lightweight nodes.
        for pid in e.get("paper_ids", []):
            add_node(str(pid), "paper")

    # Connect entity evidence to its provenance paper when possible.
    for edge in list(edges):
        pid = edge.get("paper_id")
        if pid:
            add_node(pid, "paper")
            edges.append(
                {
                    "source": edge["source"],
                    "relation": "PROVENANCE",
                    "target": pid,
                    "paper_id": pid,
                }
            )

    # Deduplicate.
    seen_edges = set()
    dedup_edges = []
    for edge in edges:
        key = (
            edge.get("source"),
            edge.get("relation"),
            edge.get("target"),
            edge.get("paper_id"),
        )
        if key not in seen_edges:
            seen_edges.add(key)
            dedup_edges.append(edge)

    return {
        "nodes": list(nodes.values()),
        "edges": dedup_edges,
    }


@app.get("/api/health")
def health() -> Dict[str, Any]:
    """返回 Stage 2 服务、语料、图谱和可用方法的状态摘要。"""
    graph_nodes, graph_edges = _graph_stats()
    return {
        "status": "ok",
        "system": "Stage2 Evidence-Centric HybridRAG",
        "papers": _paper_count(),
        "retrieval_chunks": _count_jsonl(
            ROOT / "data" / "stage2" / "retrieval_chunks.jsonl"
        ),
        "graph_nodes": graph_nodes,
        "graph_edges": graph_edges,
        "methods": ["vector", "graph", "hybrid", "rerank", "full"],
        "measured_results": {
            "structured_vector": 0.8734,
            "vanilla_hybrid": 0.7969,
            "hybrid_rerank": 0.8250,
            "claim_support_rate": 0.799,
        },
    }


@app.post("/api/ask")
def ask(request: AskRequest) -> Dict[str, Any]:
    """运行 Stage 2 问答并返回答案、证据、图结构和延迟信息。"""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    verify = bool(request.verify and request.method == "full")

    try:
        # Before-rerank evidence is useful for visualization.
        before_raw = []
        if request.method in {"rerank", "full"}:
            before_raw = SYSTEM.retrieve(query, "hybrid")

        t0 = time.perf_counter()
        result = SYSTEM.answer(query, request.method, verify=verify)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if not isinstance(result, dict):
            result = _plain(result)

        answer = result.get("answer", "") if isinstance(result, dict) else ""

        raw_evidence = []
        if isinstance(result, dict):
            raw_evidence = (
                result.get("evidence")
                or result.get("evidences")
                or result.get("contexts")
                or []
            )

        # Fall back to retrieval when the pipeline's answer() does not expose evidence.
        if not raw_evidence:
            raw_evidence = SYSTEM.retrieve(query, request.method)

        evidence = [
            _evidence_to_dict(e, i + 1)
            for i, e in enumerate(raw_evidence)
        ]
        before_evidence = [
            _evidence_to_dict(e, i + 1)
            for i, e in enumerate(before_raw)
        ]

        checks = {}
        timings = {}
        if isinstance(result, dict):
            checks = result.get("checks") or {}
            timings = (
                result.get("timings")
                or result.get("latency")
                or {}
            )

        graph = _graph_from_evidence(
            before_evidence if before_evidence else evidence
        )

        return {
            "query": query,
            "method": request.method,
            "verify_requested": verify,
            "answer": answer,
            "evidence": evidence,
            "before_evidence": before_evidence,
            "graph": graph,
            "checks": _plain(checks),
            "timings": _plain(timings),
            "request_latency_ms": round(elapsed_ms, 2),
        }

    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Stage2 pipeline execution failed: {exc}",
        ) from exc


# Serve the new frontend from the same FastAPI process.
if FRONTEND_DIR.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(FRONTEND_DIR)),
        name="assets",
    )


@app.get("/")
def frontend() -> FileResponse:
    """返回 Stage 2 前端首页，文件缺失时给出 404。"""
    index = FRONTEND_DIR / "index.html"
    if not index.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Frontend not found: {index}",
        )
    return FileResponse(index)
