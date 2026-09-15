from collections import Counter

from .utils import normalize_key, normalize_text


# 可按项目论文继续补充。
# 先覆盖 GNN 领域常见别名，避免 "Graph Convolutional Network" / "GCN" 形成两个节点。
ALIASES = {
    normalize_key("Graph Convolutional Network"): "GCN",
    normalize_key("Graph Convolutional Networks"): "GCN",
    normalize_key("Variational Graph Auto-Encoder"): "VGAE",
    normalize_key("Variational Graph Auto-Encoders"): "VGAE",
    normalize_key("Graph Attention Network"): "GAT",
    normalize_key("Graph Attention Networks"): "GAT",
    normalize_key("Graph Isomorphism Network"): "GIN",
    normalize_key("Graph Isomorphism Networks"): "GIN",
    normalize_key("Simplifying Graph Convolutional Networks"): "SGC",
    normalize_key("Approximate Personalized Propagation of Neural Predictions"): "APPNP",
    normalize_key("Link Prediction"): "Link Prediction"
}


def canonicalize_entity(name: str) -> str:
    """将实体别名规范化为知识图谱使用的统一名称。"""
    name = normalize_text(name)
    return ALIASES.get(normalize_key(name), name)


def resolve_entities(triples: list[dict]) -> list[dict]:
    """
    第一阶段 Knowledge Improvement：
    - alias normalization
    - duplicate triple removal
    - 保留 provenance metadata
    """
    normalized = []

    for triple in triples:
        t = dict(triple)
        t["source"] = canonicalize_entity(t["source"])
        t["target"] = canonicalize_entity(t["target"])
        normalized.append(t)

    seen = set()
    deduplicated = []

    for t in normalized:
        key = (
            normalize_key(t["source"]),
            t["relation"],
            normalize_key(t["target"]),
            t.get("paper_id", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(t)

    return deduplicated
