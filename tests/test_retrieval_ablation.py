from stage2.schema import Evidence
from scripts.run_retrieval_ablation import direct_select, evidence_key, pool_hash


def test_direct_selection_fills_exact_budget():
    """验证直接证据选择会填满预算并保留图证据。"""
    vectors = [Evidence(f"V{i}", "vector", str(i)) for i in range(6)]
    graph = [Evidence("G1", "triple", "g")]
    selected = direct_select(vectors, graph, 6)
    assert len(selected) == 6
    assert graph[0] in selected


def test_candidate_pool_identity_is_order_sensitive_and_stable():
    """验证候选池摘要可复现且能区分证据顺序变化。"""
    vector = ["vector:a", "vector:b"]
    graph = ["triple:x"]
    assert pool_hash(vector, graph) == pool_hash(list(vector), list(graph))
    assert pool_hash(vector, graph) != pool_hash(list(reversed(vector)), graph)


def test_path_evidence_key_includes_all_provenance():
    """验证路径证据键包含全部论文来源和关系序列。"""
    evidence = Evidence(
        "P1", "path", "a-b-c", paper_id="p1", paper_ids=["p1", "p2"],
        metadata={"nodes": ["a", "b", "c"], "relations": ["USES", "EXTENDS"]},
    )
    key = evidence_key(evidence)
    assert "p1,p2" in key
    assert "USES>EXTENDS" in key
