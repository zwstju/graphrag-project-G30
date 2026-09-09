from __future__ import annotations

import re
from typing import Dict, List


ENTITY_ALIASES = {
    "GCN": ["gcn", "graph convolutional network", "图卷积网络"],
    "VGAE": ["vgae", "variational graph auto-encoder", "variational graph autoencoder"],
    "GAT": ["gat", "graph attention network", "图注意力网络"],
    "GraphSAGE": ["graphsage", "graph sage"],
    "GIN": ["gin", "graph isomorphism network"],
    "DeepWalk": ["deepwalk"],
    "node2vec": ["node2vec", "node 2 vec"],
    "MPNN": ["mpnn", "message passing neural network"],
    "DiffPool": ["diffpool", "differentiable pooling"],
    "SGC": ["sgc", "simple graph convolution"],
    "APPNP": ["appnp", "personalized propagation of neural predictions"],
    "Cluster-GCN": ["cluster-gcn", "cluster gcn"],
    "Cora": ["cora"],
    "Citeseer": ["citeseer", "cite seer"],
    "Pubmed": ["pubmed", "pub med"],
    "Link Prediction": ["link prediction", "链接预测"],
    "Node Classification": ["node classification", "节点分类"],
    "Graph Classification": ["graph classification", "图分类"],
}


def _contains_alias(text: str, alias: str) -> bool:
    # Word-boundary matching for ASCII aliases; substring matching for CJK text.
    if re.search(r"[\u4e00-\u9fff]", alias):
        return alias in text
    pattern = r"(?<![A-Za-z0-9])" + re.escape(alias) + r"(?![A-Za-z0-9])"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def extract_entities(query: str) -> List[str]:
    found = []
    for canonical, aliases in ENTITY_ALIASES.items():
        if any(_contains_alias(query, alias) for alias in aliases):
            found.append(canonical)
    return found


def classify_intent(query: str, entities: List[str]) -> str:
    q = query.lower()

    summary_keywords = [
        "比较", "区别", "总结", "概述", "演进", "发展", "异同",
        "compare", "difference", "overview", "summarize", "evolution",
    ]
    multihop_keywords = [
        "哪些", "同时", "路径", "经过", "基于", "用于", "之间如何联系",
        "which", "path", "based on", "used for", "connect", "multi-hop",
    ]
    relation_keywords = [
        "关系", "关联", "联系", "使用", "基于", "扩展", "提出",
        "relationship", "relation", "use", "uses", "extend", "based on",
    ]
    fact_keywords = [
        "哪年", "什么时候", "谁", "哪篇", "数据集", "任务",
        "when", "year", "who", "which paper", "dataset", "task",
    ]

    if any(k in q for k in summary_keywords):
        return "summary"

    if any(k in q for k in multihop_keywords) and len(entities) >= 2:
        return "multi_hop"

    if any(k in q for k in relation_keywords) or len(entities) >= 2:
        return "relation"

    if any(k in q for k in fact_keywords):
        return "fact"

    return "general"


def process_query(query: str) -> Dict[str, object]:
    query = query.strip()
    entities = extract_entities(query)
    return {
        "normalized_query": query,
        "entities": entities,
        "intent": classify_intent(query, entities),
    }
