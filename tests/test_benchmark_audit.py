from types import SimpleNamespace

import pytest

from scripts.audit_benchmark_llm import (
    AUDIT_METHOD,
    audit_with_retries,
    build_balanced_context,
    parse_audit_response,
    select_gold_paper_evidence,
    validate_retrieval_corpus,
)
from scripts.promote_verified_benchmark import create_llm_supported_review_set
from stage2.schema import Evidence


class FakeVectorIndex:
    def __init__(self, evidence):
        """用预设证据初始化可控的向量索引替身。"""
        self.evidence = evidence
        self.index = SimpleNamespace(ntotal=len(evidence))
        self.last_query = ""

    def search(self, query, top_k):
        """记录查询并按预设顺序返回全部测试证据。"""
        self.last_query = query
        assert top_k == len(self.evidence)
        return self.evidence


def make_evidence(paper_id: str, chunk_number: int, rank: int) -> Evidence:
    """创建带论文、分块和排名信息的审计测试证据。"""
    return Evidence(
        evidence_id=f"V{rank}",
        source_type="vector",
        content=f"supporting text {paper_id} {chunk_number}",
        paper_id=paper_id,
        title=f"Title {paper_id}",
        section="Method",
        retrieval_score=1.0 - rank / 100,
        metadata={"chunk_id": f"{paper_id}_s{chunk_number:04d}"},
    )


def test_gold_aware_selection_guarantees_per_paper_quota():
    """验证金论文感知选择为每篇论文保留独立证据配额。"""
    ranked = [
        make_evidence("p1", 1, 1),
        make_evidence("p1", 2, 2),
        make_evidence("p2", 1, 3),
        make_evidence("p1", 3, 4),
        make_evidence("p2", 2, 5),
        make_evidence("p1", 4, 6),
        make_evidence("p2", 3, 7),
        make_evidence("p2", 4, 8),
    ]
    vector_index = FakeVectorIndex(ranked)
    selected, counts = select_gold_paper_evidence(
        vector_index,
        "question",
        "candidate answer",
        ["p1", "p2"],
        top_k_per_paper=3,
    )

    assert counts == {"p1": 3, "p2": 3}
    assert [item["paper_id"] for item in selected] == ["p1"] * 3 + ["p2"] * 3
    assert "candidate answer" in vector_index.last_query
    context, chars = build_balanced_context(selected, ["p1", "p2"], 6000)
    assert "=== Gold paper: p1 ===" in context
    assert "=== Gold paper: p2 ===" in context
    assert chars["p1"] > 0 and chars["p2"] > 0


def test_references_are_rejected_from_audit_corpus():
    """验证证据审计拒绝包含 References 分块的检索语料。"""
    with pytest.raises(RuntimeError, match="References"):
        validate_retrieval_corpus(
            [{"paper_id": "p1", "section": "References", "text": "citation"}]
        )


def test_parse_audit_response_uses_separate_status_fields():
    """验证模型支持判断与支持分数被解析为独立字段。"""
    result = parse_audit_response(
        '{"llm_supported": true, "llm_support_score": 0.9, '
        '"reason": "supported", "suggested_answer": "suggestion"}'
    )
    assert result == {
        "llm_supported": True,
        "llm_support_score": 0.9,
        "reason": "supported",
        "suggested_answer": "suggestion",
    }


def test_audit_retries_empty_or_invalid_json():
    """验证审计会重试空响应和格式错误的 JSON。"""
    class FakeClient:
        def __init__(self):
            """初始化依次返回空、损坏和有效响应的迭代器。"""
            self.responses = iter(
                [
                    "",
                    '{"llm_supported": true',
                    '{"llm_supported": true, "llm_support_score": 0.9, '
                    '"reason": "supported", "suggested_answer": ""}',
                ]
            )

        def chat(self, messages, temperature, max_tokens):
            """返回下一条预设审计响应。"""
            return next(self.responses)

    result, responses, errors, transport_errors, attempts = audit_with_retries(
        FakeClient(), "prompt", 3
    )
    assert result["llm_supported"] is True
    assert len(responses) == 3
    assert len(errors) == 2
    assert transport_errors == []
    assert attempts == 3


def test_exhausted_retries_preserve_attempt_counts_and_empty_responses():
    """验证重试耗尽后仍保留尝试次数及所有空响应。"""
    class EmptyClient:
        def chat(self, messages, temperature, max_tokens):
            """模拟始终返回空内容的审计客户端。"""
            return ""

    result, responses, errors, transport_errors, attempts = audit_with_retries(
        EmptyClient(), "prompt", 3
    )
    assert result is None
    assert responses == ["", "", ""]
    assert len(errors) == 3
    assert transport_errors == []
    assert attempts == 3


def test_llm_supported_promotion_never_changes_gold_or_claims_manual_review():
    """验证 LLM 支持状态不会改写金答案或冒充人工验证。"""
    benchmark = [
        {
            "qid": "Q1",
            "gold_answer": "immutable candidate",
            "review_required": True,
            "review_status": "candidate_seed",
        }
    ]
    audits = [
        {
            "id": "Q1",
            "llm_status": "llm_evidence_supported",
            "llm_supported": True,
            "llm_support_score": 0.95,
            "suggested_answer": "different suggestion",
        }
    ]
    output = create_llm_supported_review_set(
        benchmark, audits, min_score=0.8, audit_path="audit.jsonl"
    )

    assert AUDIT_METHOD == "gold-aware evidence mining for benchmark validation"
    assert output[0]["gold_answer"] == "immutable candidate"
    assert output[0]["suggested_answer"] == "different suggestion"
    assert output[0]["review_required"] is True
    assert output[0]["review_status"] == "llm_evidence_supported"
