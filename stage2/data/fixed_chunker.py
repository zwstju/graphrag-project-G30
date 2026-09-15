"""Deterministic fixed-character control chunks for the chunking ablation."""

import json
from pathlib import Path

from stage2.data.structure_chunker import split_sections


def fixed_chunks(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[tuple[int, int, str]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
    stride = chunk_size - overlap
    output = []
    for start in range(0, len(text), stride):
        content = text[start:start + chunk_size]
        if not content:
            break
        output.append((start, start + len(content), content))
        if start + chunk_size >= len(text):
            break
    return output


def non_reference_text(raw_text: str) -> str:
    """Apply the shared no-References corpus policy before fixed slicing."""
    return "\n\n".join(
        body for section, body in split_sections(raw_text)
        if section != "References"
    )


def build_fixed_chunks(
    papers_path: str,
    output_path: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> int:
    """为全部论文生成固定窗口控制块并返回写出的块数量。"""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with Path(papers_path).open(encoding="utf-8") as source, output.open("w", encoding="utf-8") as target:
        for line in source:
            if not line.strip():
                continue
            paper = json.loads(line)
            paper_id = paper.get("paper_id") or paper.get("id") or "unknown"
            text = non_reference_text(paper.get("text", ""))
            for index, (start, end, content) in enumerate(fixed_chunks(text, chunk_size, overlap)):
                record = {
                    "paper_id": paper_id,
                    "title": paper.get("title", ""),
                    "year": paper.get("year", ""),
                    "section": "Fixed",
                    "chunk_id": f"{paper_id}_f{index:04d}",
                    "start_char": start,
                    "end_char": end,
                    "text": content,
                }
                target.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
    return count
