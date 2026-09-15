import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """调用项目配置的聊天模型，并返回模型生成的文本内容。"""
    load_dotenv(PROJECT_ROOT / ".env")

    base_url = os.getenv("LLM_API_BASE", "https://api.deepseek.com").rstrip("/")
    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL", "deepseek-chat")

    if not api_key:
        raise RuntimeError("缺少 LLM_API_KEY，请检查项目根目录 .env")

    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": messages,
            "temperature": temperature,
        },
        timeout=120.0,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
