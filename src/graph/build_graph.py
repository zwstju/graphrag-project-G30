from pathlib import Path

from .chunking import split_text
from .entity_resolution import resolve_entities
from .extractor import extract_triples, refine_chunk
from .utils import read_jsonl, write_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PAPERS_PATH = PROJECT_ROOT / "data" / "processed" / "papers.jsonl"
RAW_TRIPLES_PATH = PROJECT_ROOT / "data" / "graph" / "triples_raw.jsonl"
TRIPLES_PATH = PROJECT_ROOT / "data" / "graph" / "triples.jsonl"


def build_graph_data() -> list[dict]:
    if not PAPERS_PATH.exists():
        raise FileNotFoundError(
            f"找不到 {PAPERS_PATH}，请先准备 data/processed/papers.jsonl"
        )

    papers = read_jsonl(PAPERS_PATH)
    print(f"读取论文：{len(papers)} 篇")

    raw_triples = []

    for paper_idx, paper in enumerate(papers, 1):
        paper_id = paper.get("paper_id")
        text = paper.get("text", "")

        if not paper_id or not isinstance(text, str) or not text.strip():
            print(f"[skip] {paper_id}: 缺少 paper_id 或 text")
            continue

        chunks = split_text(text)
        print(f"[{paper_idx}/{len(papers)}] {paper_id}: {len(chunks)} chunks")

        for chunk_idx, chunk in enumerate(chunks):
            chunk_id = f"{paper_id}_{chunk_idx:03d}"

            try:
                refined = refine_chunk(paper, chunk)
                triples = extract_triples(
                    paper=paper,
                    refined_text=refined,
                    chunk_id=chunk_id,
                )
            except Exception as exc:
                print(f"  {chunk_id} 抽取失败: {exc}")
                continue

            print(f"  {chunk_id}: {len(triples)} triples")
            raw_triples.extend(triples)

    write_jsonl(RAW_TRIPLES_PATH, raw_triples)

    improved_triples = resolve_entities(raw_triples)
    write_jsonl(TRIPLES_PATH, improved_triples)

    print(f"\nraw triples: {len(raw_triples)}")
    print(f"after entity resolution/dedup: {len(improved_triples)}")
    print(f"输出：{TRIPLES_PATH}")

    return improved_triples


if __name__ == "__main__":
    build_graph_data()
