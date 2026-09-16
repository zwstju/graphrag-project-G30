from stage2.schema import Evidence
from scripts.run_cross_encoder_core69_retrieval import aggregate, dedupe_evidence


def test_dedupe_evidence_preserves_first_occurrence():
    first = Evidence("G1", "triple", "A --USES--> B", paper_id="p1", metadata={"source": "A", "relation": "USES", "target": "B"})
    duplicate = Evidence("G2", "triple", "A --USES--> B", paper_id="p1", metadata={"source": "A", "relation": "USES", "target": "B"})
    assert dedupe_evidence([first, duplicate]) == [first]


def test_aggregate_keeps_hit_and_multi_gold_recall_distinct():
    metrics = {
        method: {
            "hit_at_2": 1.0,
            "gold_paper_recall_at_2": 0.5,
        }
        for method in (
            "structured_vector",
            "vanilla_hybrid",
            "rule_rerank",
            "pretrained_cross_encoder_rerank",
        )
    }
    details = [{"qtype": "comparison", "methods": metrics}]
    summary = aggregate(details, 2)
    assert summary["Comparison"]["structured_vector"]["hit_at_2"] == 1.0
    assert summary["Comparison"]["structured_vector"]["gold_paper_recall_at_2"] == 0.5
