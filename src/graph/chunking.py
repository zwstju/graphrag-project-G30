def split_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> list[str]:
    """按指定字符长度和重叠大小切分用于图谱抽取的文本。"""
    if not text:
        return []

    if overlap >= chunk_size:
        raise ValueError("overlap 必须小于 chunk_size")

    chunks = []
    start = 0 

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks
