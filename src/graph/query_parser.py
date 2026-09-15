from .entity_resolution import canonicalize_entity
from .llm import chat
from .utils import normalize_key, parse_json_array


def find_entities_lexical(query: str, graph_nodes: list[str]) -> list[str]:
    """通过规范化字面匹配从已有图节点中查找查询实体。"""
    q = normalize_key(query)

    candidates = []

    for node in graph_nodes:
        node_key = normalize_key(node)

        if node_key and node_key in q:
            candidates.append((node, node_key))

    # 长实体优先，例如先处理 VGAE，再处理 GAE
    candidates.sort(key=lambda x: len(x[1]), reverse=True)

    matches = []
    matched_keys = []

    for node, node_key in candidates:
        # 如果当前实体只是已经匹配到的更长实体的一部分，
        # 则认为是子串误匹配，不再作为独立 seed。
        is_substring_of_existing = any(
            node_key != existing_key and node_key in existing_key
            for existing_key in matched_keys
        )

        if is_substring_of_existing:
            continue

        matches.append(node)
        matched_keys.append(node_key)

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
    """优先使用字面匹配，并在无结果时调用 LLM 选择种子实体。"""
    lexical = find_entities_lexical(query, graph_nodes)
    if lexical:
        return lexical[:5]

    return find_entities_llm(query, graph_nodes, max_entities=5)
