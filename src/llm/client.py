from __future__ import annotations

import os

import httpx
from dotenv import load_dotenv

load_dotenv()


def _env_flag(name: str, default: bool) -> bool:
    """读取布尔型环境变量，并在变量缺失时返回指定默认值。"""
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _mock_answer(prompt: str) -> str:
    """在 mock 模式下返回固定答案，便于无外部模型时验证调用链。"""
    return (
        "[MOCK LLM] Backend、Retriever、Prompt Builder 已成功形成闭环。"
        "请在 .env 中配置真实 LLM_API_BASE / LLM_API_KEY / LLM_MODEL，"
        "并将 LLM_MOCK_MODE=false 后使用真实模型生成答案。"
    )


def generate_answer(prompt: str) -> str:
    """
    Call an OpenAI-compatible /chat/completions endpoint.

    Required when LLM_MOCK_MODE=false:
      - LLM_API_BASE
      - LLM_API_KEY
      - LLM_MODEL

    Example API base:
      https://api.example.com/v1
    """
    if _env_flag("LLM_MOCK_MODE", True):
        return _mock_answer(prompt)

    api_base = os.getenv("LLM_API_BASE", "").rstrip("/")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL", "")

    if not api_base or not api_key or not model:
        raise RuntimeError(
            "LLM is not configured. Set LLM_API_BASE, LLM_API_KEY, "
            "LLM_MODEL, or enable LLM_MOCK_MODE=true."
        )

    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
    temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", "1024"))

    url = f"{api_base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()

    try:
        #return data["choices"][0]["message"]["content"].strip()
        content = data["choices"][0]["message"]["content"].strip()

        print("\n===== LLM DEBUG =====")
        print("content repr:", repr(content))
        print("content length:", len(content))
        print("finish_reason:", data["choices"][0].get("finish_reason"))
        print("model:", data.get("model"))
        print("===== END LLM DEBUG =====\n")

        return content
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Unexpected LLM response format: {data}"
        ) from exc
