from __future__ import annotations

from typing import Any, Dict, List


def format_vector_context(results: List[Dict[str, Any]]) -> str:
    if not results:
        return "(no vector evidence retrieved)"

    blocks = []
    for idx, item in enumerate(results, start=1):
        paper_id = item.get("paper_id", "unknown")
        chunk_id = item.get("chunk_id", "unknown")
        score = item.get("score", "")
        title = item.get("title")
        section = item.get("section")
        text = item.get("text", "")

        meta = [f"paper_id={paper_id}", f"chunk_id={chunk_id}"]
        if title:
            meta.append(f"title={title}")
        if section:
            meta.append(f"section={section}")
        if score != "":
            try:
                meta.append(f"score={float(score):.4f}")
            except (TypeError, ValueError):
                meta.append(f"score={score}")

        blocks.append(
            f"[Vector Evidence {idx} | {' | '.join(meta)}]\n{text}".strip()
        )

    return "\n\n".join(blocks)


def _format_path(path: Dict[str, Any]) -> str:
    nodes = path.get("nodes", [])
    relations = path.get("relations", [])
    score = path.get("score")

    if not nodes:
        return ""

    parts = [str(nodes[0])]
    for i, relation in enumerate(relations):
        if i + 1 >= len(nodes):
            break
        parts.append(f"--{relation}-->")
        parts.append(str(nodes[i + 1]))

    text = " ".join(parts)
    if score is not None:
        try:
            text += f" (score={float(score):.4f})"
        except (TypeError, ValueError):
            text += f" (score={score})"
    return text


def format_graph_context(result: Dict[str, Any]) -> str:
    if not result:
        return "(no graph evidence retrieved)"

    lines = []

    entities = result.get("entities", [])
    if entities:
        lines.append("Matched entities: " + ", ".join(map(str, entities)))

    triples = result.get("triples", [])
    for idx, triple in enumerate(triples, start=1):
        source = triple.get("source", "")
        relation = triple.get("relation", "")
        target = triple.get("target", "")
        paper_id = triple.get("paper_id", "unknown")
        lines.append(
            f"[Triple {idx} | paper_id={paper_id}] "
            f"{source} --{relation}--> {target}"
        )

    paths = result.get("paths", [])
    for idx, path in enumerate(paths, start=1):
        path_text = _format_path(path)
        if path_text:
            lines.append(f"[Path {idx}] {path_text}")

    if not lines:
        return "(no graph evidence retrieved)"

    return "\n".join(lines)


def build_prompt(
    query: str,
    vector_results: List[Dict[str, Any]],
    graph_results: Dict[str, Any],
    mode: str = "hybrid",
    query_analysis: Dict[str, Any] | None = None,
) -> str:
    if mode not in {"vector", "graph", "hybrid"}:
        raise ValueError(f"Unsupported mode: {mode}")

    analysis_text = ""
    if query_analysis:
        entities = query_analysis.get("entities", [])
        intent = query_analysis.get("intent", "unknown")
        analysis_text = (
            f"\nQuery analysis:\n"
            f"- intent: {intent}\n"
            f"- recognized entities: {', '.join(entities) if entities else '(none)'}\n"
        )

    vector_context = format_vector_context(vector_results)
    graph_context = format_graph_context(graph_results)

    if mode == "vector":
        evidence = f"Vector Evidence:\n{vector_context}"
    elif mode == "graph":
        evidence = f"Graph Evidence:\n{graph_context}"
    else:
        # This deliberately mirrors the baseline HybridRAG paper:
        # vector context first, graph context second.
        evidence = (
            f"Vector Evidence:\n{vector_context}\n\n"
            f"Graph Evidence:\n{graph_context}"
        )

    return f"""You are a research assistant for graph neural network literature.

Rules:
1. Answer the user's question using the retrieved evidence as the primary basis.
2. Do not invent facts that are not supported by the evidence.
3. If the evidence is insufficient, explicitly say that the current retrieval evidence is insufficient.
4. When possible, cite supporting paper_id values in square brackets, for example [vgae_2016].
5. Keep the answer focused on the question and distinguish direct evidence from reasonable interpretation.

Retrieval mode: {mode}
{analysis_text}
User Question:
{query}

{evidence}

Answer:
""".strip()
