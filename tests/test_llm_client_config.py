from stage2.llm.client import LLMClient


def test_llm_max_tokens_override(monkeypatch):
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    client = LLMClient(key="test")
    assert client.effective_max_tokens(300) == 8192


def test_llm_max_tokens_default(monkeypatch):
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    client = LLMClient(key="test")
    assert client.effective_max_tokens(300) == 300
