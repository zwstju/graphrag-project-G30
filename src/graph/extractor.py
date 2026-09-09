from .llm import chat
from .schema import ENTITY_TYPES, RELATION_TYPES
from .utils import normalize_text, parse_json_array


def refine_chunk(paper: dict, text: str) -> str:
    """
    HybridRAG 论文式两层 LLM 链中的第一层：
    把原始 chunk 变成更利于知识抽取的精炼表示。
    """
    prompt = f"""
You are refining a research-paper passage for knowledge graph extraction.

Preserve:
- model names
- datasets
- tasks
- methods
- authors
- citations
- explicit technical relationships

Remove:
- formatting noise
- duplicated wording
- irrelevant boilerplate

Do not add facts not supported by the passage.

Paper title: {paper.get("title", "")}
Year: {paper.get("year", "")}

Passage:
{text}

Return only the refined passage.
""".strip()

    return normalize_text(
        chat(
            [
                {
                    "role": "system",
                    "content": "You refine scientific text without inventing information.",
                },
                {"role": "user", "content": prompt},
            ]
        )
    )


def extract_triples(paper: dict, refined_text: str, chunk_id: str) -> list[dict]:
    """
    第二层 LLM：
    从精炼文本抽取结构化实体和关系。
    """
    prompt = f"""
Extract knowledge-graph triples from this GNN research-paper passage.

Allowed entity types:
{sorted(ENTITY_TYPES)}

Allowed relations:
{sorted(RELATION_TYPES)}

Requirements:
1. Extract only relations explicitly supported by the text.
2. Prefer canonical model/dataset names.
3. Do not invent citations or relations.
4. Avoid duplicates.
5. Return ONLY a JSON array.
6. Each object must contain:
   source, source_type, relation, target, target_type

Paper metadata:
paper_id: {paper.get("paper_id", "")}
title: {paper.get("title", "")}
year: {paper.get("year", "")}

Refined passage:
{refined_text}

Example:
[
  {{
    "source": "VGAE",
    "source_type": "Model",
    "relation": "USES",
    "target": "GCN",
    "target_type": "Model"
  }}
]
""".strip()

    raw = chat(
        [
            {
                "role": "system",
                "content": "You are a precise knowledge graph extraction system. Return valid JSON only.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    rows = parse_json_array(raw)
    triples = []

    for row in rows:
        if not isinstance(row, dict):
            continue

        source = normalize_text(row.get("source"))
        target = normalize_text(row.get("target"))
        source_type = normalize_text(row.get("source_type"))
        target_type = normalize_text(row.get("target_type"))
        relation = normalize_text(row.get("relation")).upper()

        if not source or not target:
            continue
        if source_type not in ENTITY_TYPES or target_type not in ENTITY_TYPES:
            continue
        if relation not in RELATION_TYPES:
            continue

        triples.append(
            {
                "source": source,
                "source_type": source_type,
                "relation": relation,
                "target": target,
                "target_type": target_type,
                "paper_id": paper.get("paper_id", ""),
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "chunk_id": chunk_id,
            }
        )

    return triples
