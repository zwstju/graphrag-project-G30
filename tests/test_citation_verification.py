from scripts.run_citation_verification import text_sha256
from stage2.verify.citation import split_atomic_claims


def test_answer_hash_is_stable_and_content_sensitive():
    assert text_sha256("answer") == text_sha256("answer")
    assert text_sha256("answer") != text_sha256("answer changed")


def test_deterministic_claim_split_removes_citations_and_bullets():
    answer = "- 第一条事实。[paper_a]\n2. Second claim! [paper_b]"
    assert split_atomic_claims(answer) == ["第一条事实", "Second claim"]
