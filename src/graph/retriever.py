from collections import deque
from pathlib import Path

import networkx as nx

from .query_parser import find_seed_entities
from .utils import normalize_key, read_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRIPLES_PATH = PROJECT_ROOT / "data" / "graph" / "triples.jsonl"

# 原计划第一阶段 GraphRAG baseline：固定 1-hop。
# 第二阶段 Query-Adaptive Graph Retrieval 再改动态 1-2 hop。
DEFAULT_DEPTH = 1

_graph = None
_triples = None


def _load_graph():
    global _graph, _triples

    if _graph is not None:
        return

    if not TRIPLES_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {TRIPLES_PATH}，请先运行 python -m src.graph.build_graph"
        )

    triples = read_jsonl(TRIPLES_PATH)
    graph = nx.MultiDiGraph()

    for t in triples:
        source = t["source"]
        target = t["target"]

        graph.add_node(source, type=t.get("source_type", ""))
        graph.add_node(target, type=t.get("target_type", ""))

        graph.add_edge(
            source,
            target,
            relation=t["relation"],
            paper_id=t.get("paper_id", ""),
            title=t.get("title", ""),
            year=t.get("year"),
            chunk_id=t.get("chunk_id", ""),
        )

    _graph = graph
    _triples = triples

    print(
        f"Graph loaded: {_graph.number_of_nodes()} nodes, "
        f"{_graph.number_of_edges()} edges"
    )


def _expand_nodes(seeds: list[str], depth: int) -> set[str]:
    """
    用无向视图扩展邻居，从而同时覆盖入边和出边。
    返回的 triples 仍保留原始有向关系。
    """
    undirected = _graph.to_undirected()
    visited = set(seeds)
    queue = deque((seed, 0) for seed in seeds)

    while queue:
        node, d = queue.popleft()

        if d >= depth:
            continue

        for neighbor in undirected.neighbors(node):
            if neighbor in visited:
                continue

            visited.add(neighbor)
            queue.append((neighbor, d + 1))

    return visited


def _score_triple(query: str, triple: dict, seeds: list[str]) -> float:
    """
    第一阶段规则打分：
    - seed entity 命中优先
    - query 中出现 source/target 优先
    - relation 关键词轻量加分
    """
    q = normalize_key(query)
    score = 0.0

    if triple["source"] in seeds:
        score += 3.0
    if triple["target"] in seeds:
        score += 3.0

    if normalize_key(triple["source"]) in q:
        score += 1.0
    if normalize_key(triple["target"]) in q:
        score += 1.0

    relation_words = triple["relation"].replace("_", " ").lower().split()
    q_lower = query.lower()
    score += sum(0.25 for w in relation_words if w in q_lower)

    return score


def _edge_relation(a: str, b: str) -> str:
    if _graph.has_edge(a, b):
        data = _graph.get_edge_data(a, b)
        key = next(iter(data))
        return data[key].get("relation", "RELATED_TO")

    if _graph.has_edge(b, a):
        data = _graph.get_edge_data(b, a)
        key = next(iter(data))
        return data[key].get("relation", "RELATED_TO")

    return "RELATED_TO"


def _collect_paths(seeds: list[str], max_paths: int = 5) -> list[dict]:
    """
    seeds 之间存在连接时返回最短关系路径。
    用于关系/多跳问题的可解释证据。
    """
    if len(seeds) < 2:
        return []

    undirected = _graph.to_undirected()
    paths = []

    for i in range(len(seeds)):
        for j in range(i + 1, len(seeds)):
            try:
                nodes = nx.shortest_path(
                    undirected,
                    source=seeds[i],
                    target=seeds[j],
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            # 第一阶段避免返回过长路径
            if len(nodes) > 5:
                continue

            relations = [
                _edge_relation(a, b)
                for a, b in zip(nodes, nodes[1:])
            ]

            paths.append(
                {
                    "nodes": nodes,
                    "relations": relations,
                    "score": 1.0 / max(1, len(nodes) - 1),
                }
            )

    paths.sort(key=lambda p: p["score"], reverse=True)
    return paths[:max_paths]


def retrieve_graph(query: str, top_k: int = 5) -> dict:
    """
    对外统一接口：

    {
        "entities": [],
        "triples": [],
        "paths": [],
        "paper_ids": []
    }
    """
    query = query.strip()

    if not query:
        raise ValueError("query cannot be empty")

    if top_k <= 0:
        return {
            "entities": [],
            "triples": [],
            "paths": [],
            "paper_ids": [],
        }

    _load_graph()

    seeds = find_seed_entities(query, list(_graph.nodes))

    if not seeds:
        return {
            "entities": [],
            "triples": [],
            "paths": [],
            "paper_ids": [],
        }

    nodes = _expand_nodes(seeds, DEFAULT_DEPTH)

    candidate_triples = [
        t
        for t in _triples
        if t["source"] in nodes and t["target"] in nodes
    ]

    candidate_triples.sort(
        key=lambda t: _score_triple(query, t, seeds),
        reverse=True,
    )

    triples = candidate_triples[:top_k]

    entity_names = set(seeds)
    for t in triples:
        entity_names.add(t["source"])
        entity_names.add(t["target"])

    entities = [
        {
            "name": node,
            "type": _graph.nodes[node].get("type", ""),
            "is_seed": node in seeds,
        }
        for node in sorted(entity_names)
    ]

    paths = _collect_paths(seeds, max_paths=top_k)

    paper_ids = sorted(
        {
            t.get("paper_id")
            for t in triples
            if t.get("paper_id")
        }
    )

    return {
        "entities": entities,
        "triples": triples,
        "paths": paths,
        "paper_ids": paper_ids,
    }


if __name__ == "__main__":
    import json

    queries = [
        "VGAE 和 GCN 有什么关系？",
        "GCN 和 Link Prediction 有什么关联？",
    ]

    for q in queries:
        print(f"\nQUERY: {q}")
        print(
            json.dumps(
                retrieve_graph(q, top_k=5),
                ensure_ascii=False,
                indent=2,
            )
        )
