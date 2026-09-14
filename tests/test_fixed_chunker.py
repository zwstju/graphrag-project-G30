import json

from stage2.data.fixed_chunker import build_fixed_chunks, fixed_chunks


def test_fixed_chunks_have_exact_stride_and_overlap():
    chunks = fixed_chunks("x" * 2500, chunk_size=1200, overlap=200)
    assert [(start, end) for start, end, _ in chunks] == [
        (0, 1200), (1000, 2200), (2000, 2500)
    ]
    assert chunks[0][2][-200:] == chunks[1][2][:200]


def test_fixed_builder_excludes_detected_references(tmp_path):
    papers = tmp_path / "papers.jsonl"
    output = tmp_path / "chunks.jsonl"
    paper = {
        "paper_id": "p1", "title": "Paper", "year": 2020,
        "text": "Introduction\n" + "method text " * 120 + "\nReferences\nforbidden citation " * 100,
    }
    papers.write_text(json.dumps(paper) + "\n", encoding="utf-8")
    build_fixed_chunks(str(papers), str(output), 1200, 200)
    records = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert records
    assert all(record["section"] == "Fixed" for record in records)
    assert all("forbidden citation" not in record["text"] for record in records)
