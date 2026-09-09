import os

os.environ["ALLOW_MOCK_RETRIEVERS"] = "true"
os.environ["LLM_MOCK_MODE"] = "true"

from src.hybrid.pipeline import answer_query


def test_hybrid_pipeline_contract():
    result = answer_query(
        "VGAE 和 GCN 有什么关系？",
        mode="hybrid",
    )

    assert result["query"]
    assert result["mode"] == "hybrid"
    assert isinstance(result["answer"], str)
    assert isinstance(result["vector_results"], list)
    assert isinstance(result["graph_results"], dict)
    assert isinstance(result["references"], list)

    for key in ["entities", "triples", "paths", "paper_ids"]:
        assert key in result["graph_results"]


def test_vector_mode_has_empty_graph_contract():
    result = answer_query("What is VGAE?", mode="vector")
    assert result["vector_results"]
    assert result["graph_results"] == {
        "entities": [],
        "triples": [],
        "paths": [],
        "paper_ids": [],
    }


def test_graph_mode_has_empty_vector_contract():
    result = answer_query("VGAE 和 GCN 有什么关系？", mode="graph")
    assert result["vector_results"] == []
    assert result["graph_results"]["triples"]
