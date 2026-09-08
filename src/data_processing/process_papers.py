"""Offline PDF -> shared paper JSONL. Run from any working directory."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parents[2]
LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl", "\u00a0": " "})


def clean_page(text, page_number, vocabulary):
    text = text.translate(LIGATURES).replace("\x00", "").replace("\u00ad", "")
    lines = text.splitlines()
    kept = []
    for index, line in enumerate(lines):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line.startswith("arXiv:") or line.startswith("Published as a conference paper at"):
            continue
        # Only remove a matching page number at the page boundary, not equation values.
        if line == str(page_number) and (index < 2 or index >= len(lines) - 3):
            continue
        kept.append(line)
    text = "\n".join(kept)
    # Join hyphenated line breaks only when the joined word occurs elsewhere in this PDF.
    # Otherwise preserve the hyphen: e.g. graph-structured and Weisfeiler-Lehman.
    def join_word(match):
        left, right = match.group(1), match.group(2)
        return left + ("" if (left + right).lower() in vocabulary else "-") + right
    text = re.sub(r"([A-Za-z]+)-\n([a-z]+)\b", join_word, text)
    # Unwrap prose conservatively; retain heading/equation/paragraph boundaries.
    lines = text.splitlines()
    result = []
    for line in lines:
        if (result and len(result[-1]) >= 60 and line and line[0].islower()
                and result[-1][-1:] not in ".:;!?" and result[-1][-1:].isalpha()):
            result[-1] += " " + line
        else:
            result.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(result)).strip()


def prepare_page(text, item, page_number):
    """Repairs observed in the supplied seven PDFs; keeps the original pipeline for other IDs."""
    if item["paper_id"] == "deepwalk_2014":
        text = re.sub(r"Deep-?\s*W\s*alk", "DeepWalk", text)
        text = re.sub(r"DeepWalk(?=[a-z])", "DeepWalk ", text)
    if item["paper_id"] == "node2vec_2016":
        text = text.replace("ofnode2vec", "of node2vec")
    if item["paper_id"] == "sgc_2019":
        text = re.sub(r"<latexit\b[^>]*>.*?</latexit>", "", text, flags=re.DOTALL)
    if page_number > 1 and item["paper_id"] in {"mpnn_2017", "sgc_2019"}:
        lines = text.splitlines()
        lines = [line for index, line in enumerate(lines)
                 if not (index < 3 and line.strip() == item["title"])]
        text = "\n".join(lines)
    if page_number == 1 and item["paper_id"] == "diffpool_2018":
        text = text.replace("Preprint. Work in progress.", "")
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    data = args.data_dir
    manifest = json.loads((data / "papers_manifest.json").read_text(encoding="utf-8"))
    if not manifest or len({m["paper_id"] for m in manifest}) != len(manifest):
        raise ValueError("Manifest must contain records with unique paper_id values")
    records, report, page_records = [], [], []
    for item in manifest:
        for key in ("paper_id", "title", "filename", "body_start"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Invalid {key}")
        if type(item["year"]) is not int or not isinstance(item["authors"], list) or not item["authors"] or not all(isinstance(a, str) and a.strip() for a in item["authors"]):
            raise ValueError("Invalid year or authors")
        pdf = data / "raw" / item["filename"]
        reader = PdfReader(pdf)
        raw_pages = [page.extract_text() or "" for page in reader.pages]
        if any(len(t.strip()) < 20 for t in raw_pages):
            raise ValueError(f"{pdf.name}: blank/image-only page; inspect before processing")
        vocabulary = set(re.findall(r"\b[a-zA-Z]{3,}\b", "\n".join(raw_pages).translate(LIGATURES).lower()))
        pages = []
        for number, raw in enumerate(raw_pages, 1):
            cleaned = clean_page(prepare_page(raw, item, number), number, vocabulary)
            if number == 1:
                marker = item["body_start"]
                if marker not in cleaned:
                    raise ValueError(f"{pdf.name}: body start not found")
                cleaned = cleaned[cleaned.index(marker):]
            if not cleaned:
                raise ValueError(f"{pdf.name}: empty cleaned page {number}")
            pages.append(cleaned)
            page_records.append({"paper_id": item["paper_id"], "page": number, "text": cleaned})
        text = "\n\n".join(pages)
        if len(text) < 1000 or "\ufffd" in text:
            raise ValueError(f"{pdf.name}: suspicious extracted text")
        record = {key: item[key] for key in ("paper_id", "title", "authors", "year")}
        record["text"] = text
        records.append(record)
        report.append({"paper_id": item["paper_id"], "filename": pdf.name,
                       "source_url": item["source_url"], "sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(),
                       "pages": len(pages), "characters": len(text),
                       "page_characters": [len(t) for t in pages],
                       "replacement_characters": text.count("\ufffd"),
                       "limitations": "Text layer only; equations/tables/images are not reconstructed. References and appendices retained."})
    # All inputs must succeed before writing the shared corpus.
    output = data / "processed"
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in (("papers.jsonl", records), ("paper_pages.jsonl", page_records)):
        (output / name).write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    (output / "quality_report.json").write_text(json.dumps({"paper_count": len(records), "total_pages": sum(r["pages"] for r in report), "papers": report}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for row in report:
        print(f"{row['paper_id']}: {row['pages']} pages, {row['characters']} characters")


if __name__ == "__main__":
    main()