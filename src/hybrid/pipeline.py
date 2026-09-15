from __future__ import annotations

from typing import Any, Dict, List

from src.hybrid.prompt import build_prompt
from src.llm.client import generate_answer
from src.query.processor import process_query
from src.retrieval.adapters import retrieve_graph, retrieve_vector


def empty_graph_result() -> Dict[str, List[Any]]:
    """创建符合 Graph 检索接口约定的空结果结构。"""
    return {
        "entities": [],
        "triples": [],
        "paths": [],
        "paper_ids": [],
    }


def retrieve_hybrid(
    query: str,
    vector_top_k: int = 5,
    graph_top_k: int = 5,
) -> Dict[str, Any]:
    """分别执行向量与图检索，并返回未经融合的两组候选结果。"""
    return {
        "vector_results": retrieve_vector(query, top_k=vector_top_k),
        "graph_results": retrieve_graph(query, top_k=graph_top_k),
    }


def _collect_references(
    vector_results: List[Dict[str, Any]],
    graph_results: Dict[str, Any],
) -> List[Dict[str, str]]:
    """从向量和图证据中汇总去重后的论文引用信息。"""
    refs: Dict[str, Dict[str, str]] = {}

    for item in vector_results:
        paper_id = item.get("paper_id")
        if not paper_id:
            continue
        ref = {"paper_id": str(paper_id)}
        if item.get("title"):
            ref["title"] = str(item["title"])
        refs[str(paper_id)] = ref

    for paper_id in graph_results.get("paper_ids", []):
        paper_id = str(paper_id)
        refs.setdefault(paper_id, {"paper_id": paper_id})

    # Also recover paper_ids directly from triples in case the graph module
    # forgot to aggregate them at top level.
    for triple in graph_results.get("triples", []):
        paper_id = triple.get("paper_id")
        if paper_id:
            paper_id = str(paper_id)
            refs.setdefault(paper_id, {"paper_id": paper_id})

    return list(refs.values())


def answer_query(
    query: str,
    mode: str = "hybrid",
    vector_top_k: int = 5,
    graph_top_k: int = 5,
) -> Dict[str, Any]:
    """按指定检索模式组织证据、构建提示词并生成最终回答。"""
    query = query.strip()
    if not query:
        raise ValueError("query must not be empty")

    if mode not in {"vector", "graph", "hybrid"}:
        raise ValueError(
            f"Unsupported mode '{mode}'. Expected vector, graph, or hybrid."
        )

    analysis = process_query(query)

    vector_results: List[Dict[str, Any]] = []
    graph_results: Dict[str, Any] = empty_graph_result()

    if mode == "vector":
        vector_results = retrieve_vector(query, top_k=vector_top_k)

    elif mode == "graph":
        graph_results = retrieve_graph(query, top_k=graph_top_k)

    else:
        retrieved = retrieve_hybrid(
            query,
            vector_top_k=vector_top_k,
            graph_top_k=graph_top_k,
        )
        vector_results = retrieved["vector_results"]
        graph_results = retrieved["graph_results"]

    prompt = build_prompt(
        query=query,
        vector_results=vector_results,
        graph_results=graph_results,
        mode=mode,
        query_analysis=analysis,
    )

    if not vector_results and not graph_results.get("triples") and not graph_results.get("paths"):
        answer = "当前检索模块没有返回足够证据，无法基于外部知识可靠回答该问题。"
    else:
        answer = generate_answer(prompt)

    return {
        "query": query,
        "mode": mode,
        "answer": answer,
        "vector_results": vector_results,
        "graph_results": graph_results,
        "references": _collect_references(vector_results, graph_results),
    }
