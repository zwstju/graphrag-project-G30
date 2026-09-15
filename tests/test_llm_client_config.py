from stage2.llm.client import LLMClient


def test_llm_max_tokens_override(monkeypatch):
    """验证环境变量可以覆盖单次调用请求的最大生成长度。"""
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")
    client = LLMClient(key="test")
    assert client.effective_max_tokens(300) == 8192


def test_llm_max_tokens_default(monkeypatch):
    """验证未配置环境变量时沿用调用方指定的最大生成长度。"""
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    client = LLMClient(key="test")
    assert client.effective_max_tokens(300) == 300
