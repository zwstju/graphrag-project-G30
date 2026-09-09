from .entity_resolution import canonicalize_entity
from .llm import chat
from .utils import normalize_key, parse_json_array


def find_entities_lexical(query: str, graph_nodes: list[str]) -> list[str]:
    q = normalize_key(query)
    matches = []

    for node in graph_nodes:
        node_key = normalize_key(node)
        if node_key and node_key in q:
            matches.append(node)

    matches.sort(key=len, reverse=True)
    return list(dict.fromkeys(matches))


def find_entities_llm(query: str, graph_nodes: list[str], max_entities: int = 5) -> list[str]:
    """
    当字面匹配找不到实体时，用 LLM 在已有图节点中做 entity linking。
    不允许 LLM 创造不存在于图中的 seed node。
    """
    node_text = "\n".join(f"- {n}" for n in graph_nodes[:1000])

    prompt = f"""
Given the user query and the list of existing knowledge-graph nodes,
select the nodes that are explicitly or semantically referred to by the query.

Return ONLY a JSON array of node names.
Do not invent node names.
Return at most {max_entities} nodes.

Query:
{query}

Existing graph nodes:
{node_text}
""".strip()

    raw = chat(
        [
            {
                "role": "system",
                "content": "You perform entity linking against a fixed knowledge graph vocabulary.",
            },
            {"role": "user", "content": prompt},
        ]
    )

    candidates = parse_json_array(raw)
    valid = {normalize_key(n): n for n in graph_nodes}

    result = []
    for item in candidates:
        name = canonicalize_entity(str(item))
        key = normalize_key(name)
        if key in valid:
            result.append(valid[key])

    return list(dict.fromkeys(result))[:max_entities]


def find_seed_entities(query: str, graph_nodes: list[str]) -> list[str]:
    lexical = find_entities_lexical(query, graph_nodes)
    if lexical:
        return lexical[:5]

    return find_entities_llm(query, graph_nodes, max_entities=5)
