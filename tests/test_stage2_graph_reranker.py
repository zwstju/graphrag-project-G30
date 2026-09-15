import json

from stage2.retrieval.graph import GraphRetriever
from stage2.rerank.evidence_reranker import EvidenceReranker, qtype, tokens
from stage2.schema import Evidence


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        """为所有输入生成相同的确定性二维测试向量。"""
        import numpy as np
        return np.tile(np.array([[1.0, 0.0]], dtype="float32"), (len(texts), 1))


def make_graph(tmp_path, triples):
    """将测试三元组写入临时文件并创建图检索器。"""
    path = tmp_path / "triples.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in triples), encoding="utf-8")
    return GraphRetriever(str(path))


def test_entity_matching_does_not_seed_sgc_from_uses_gcn(tmp_path):
    """验证实体匹配不会因边关系而错误地把 SGC 识别为查询实体。"""
    graph = make_graph(tmp_path, [
        {"source": "SGC", "relation": "SIMPLIFIES", "target": "GCN", "paper_id": "sgc"},
        {"source": "VGAE", "relation": "USES", "target": "GCN", "paper_id": "vgae"},
    ])
    seeds = graph.find_seeds("Which model uses GCN as an encoder?")
    assert "GCN" in seeds
    assert "SGC" not in seeds


def test_graph_top_k_is_strict_and_noise_is_filtered(tmp_path):
    """验证图检索严格限制结果数并过滤引用类噪声边。"""
    graph = make_graph(tmp_path, [
        {"source": "VGAE", "relation": "USES", "target": "GCN", "paper_id": "p1"},
        {"source": "VGAE", "relation": "CITES", "target": "GCN", "paper_id": "noise"},
        {"source": "VGAE", "relation": "PROPOSES", "target": "latent model", "paper_id": "p2"},
    ])
    result = graph.search("How does VGAE use GCN?", top_k=1)
    assert len(result) == 1
    assert all("CITES" not in item.content for item in result)


def test_parallel_edge_choice_is_relevance_scored_and_deterministic(tmp_path):
    """验证平行边按查询相关性稳定选择最合适的关系。"""
    graph = make_graph(tmp_path, [
        {"source": "VGAE", "relation": "RELATED_TO", "target": "GCN", "paper_id": "p2"},
        {"source": "VGAE", "relation": "USES", "target": "GCN", "paper_id": "p1"},
    ])
    choice = graph._choose_edge("Which model uses GCN?", "VGAE", "GCN")
    assert choice[-1]["relation"] == "USES"


def test_path_preserves_all_paper_ids(tmp_path):
    """验证跨边路径证据完整保留所有论文来源。"""
    graph = make_graph(tmp_path, [
        {"source": "Alpha", "relation": "USES", "target": "Bridge", "paper_id": "p1"},
        {"source": "Bridge", "relation": "EXTENDS", "target": "Gamma", "paper_id": "p2"},
    ])
    result = graph.search("How are Alpha and Gamma related?", top_k=10)
    path = next(item for item in result if item.source_type == "path")
    assert path.paper_ids == ["p1", "p2"]
    assert path.metadata["paper_ids"] == ["p1", "p2"]


def test_qtype_inference_and_mixed_tokens():
    """验证中英混合查询的题型推断和关键词切分。"""
    assert qtype("GCN 和 GAT 有什么区别？") == "comparison"
    assert qtype("哪个模型既使用 GCN 编码器，又重构边？") == "multi_hop"
    assert qtype("VGAE 和 GCN 有什么关系？") == "relation"
    query_tokens = tokens("VGAE 和 GCN 有什么关系？")
    assert {"vgae", "gcn"} <= query_tokens


def test_relation_part_uses_actual_relation_content():
    """验证重排器的关系分数使用证据中的实际关系类型。"""
    reranker = EvidenceReranker(FakeEmbedder())
    uses = Evidence("G1", "triple", "VGAE --USES--> GCN", metadata={"relation": "USES"})
    cites = Evidence("G2", "triple", "VGAE --CITES--> GCN", metadata={"relation": "CITES"})
    assert reranker._relation("Which model uses GCN?", uses) > reranker._relation("Which model uses GCN?", cites)


def test_reranker_does_not_penalize_valid_same_paper_evidence():
    """验证重排器不会无故淘汰同一论文中的不同有效证据。"""
    reranker = EvidenceReranker(FakeEmbedder())
    evidence = [
        Evidence(f"V{i}", "vector", f"distinct {i}", paper_id="gold", retrieval_score=1 - i / 10)
        for i in range(4)
    ]
    selected = reranker.rerank("What is the method?", evidence, top_k=4)
    assert len(selected) == 4
    assert all(item.paper_id == "gold" for item in selected)
