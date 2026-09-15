import json
import re
from pathlib import Path


def normalize_text(text: str) -> str:
    """把输入转为字符串并合并多余空白。"""
    return re.sub(r"\s+", " ", str(text or "").strip())


def normalize_key(text: str) -> str:
    """生成仅保留字母、数字和中文字符的大小写无关比较键。"""
    text = normalize_text(text).casefold()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", "", text)
    return text


def strip_markdown_fence(text: str) -> str:
    """移除模型输出外围可能存在的 Markdown 代码围栏。"""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_json_array(text: str) -> list:
    """从纯文本或混合文本中尽可能解析出 JSON 数组。"""
    text = strip_markdown_fence(text)

    try:
        data = json.loads(text)
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        pass

    left = text.find("[")
    right = text.rfind("]")
    if left == -1 or right == -1 or right <= left:
        return []

    try:
        data = json.loads(text[left:right + 1])
        return data if isinstance(data, list) else []
    except json.JSONDecodeError:
        return []


def read_jsonl(path: Path) -> list[dict]:
    """读取 JSONL 文件并在格式错误时报告准确行号。"""
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path} 第 {line_no} 行不是合法 JSON") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """将字典序列以 UTF-8 JSONL 格式写入目标路径。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
