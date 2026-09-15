import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# We only keep broad scientific-paper sections for reranking.
# This deliberately ignores most short/capitalized lines because PDF extraction
# often turns table cells and captions into fake headings.
SECTION_PATTERNS = [
    ("Abstract", re.compile(r"^abstract$", re.I)),
    ("Introduction", re.compile(r"^(?:introduction|overview)$", re.I)),
    ("Related Work", re.compile(r"^(?:related work|related works|prior work|literature review)$", re.I)),
    ("Background", re.compile(r"^(?:background|preliminaries|preliminary|notation)$", re.I)),
    ("Method", re.compile(
        r"^(?:method|methods|methodology|approach|model|models|architecture|architectures|"
        r"proposed method|proposed model|framework|algorithm|algorithms)$", re.I)),
    ("Experiments", re.compile(
        r"^(?:experiments?|experimental setup|evaluation|datasets?|data|training|implementation details)$", re.I)),
    ("Results", re.compile(r"^(?:results?|analysis|discussion|ablation|ablation study)$", re.I)),
    ("Conclusion", re.compile(r"^(?:conclusion|conclusions|concluding remarks)$", re.I)),
    ("Appendix", re.compile(r"^(?:appendix|supplementary material|supplement)$", re.I)),
    ("References", re.compile(r"^(?:references|bibliography)$", re.I)),
]

METHOD_HINTS = (
    "architecture", "aggregator", "message passing", "graph attention",
    "graph attentional", "propagation", "pooling", "random walk",
    "search strategy", "search strategies", "encoder", "decoder",
    "neural network", "learning algorithm", "model formulation",
)
EXPERIMENT_HINTS = (
    "experiment", "evaluation", "dataset", "classification", "link prediction",
    "training performance", "parameter sensitivity", "implementation",
)

NUMBERING_RE = re.compile(
    r"^\s*(?:(?:\d+(?:\.\d+)*)|(?:[IVXLC]+))[\.\)]?\s+",
    re.I,
)

KNOWN_OCR_REPAIRS = {
    "i ntroduction": "Introduction",
    "e xperiments": "Experiments",
    "e xperimental setup": "Experimental Setup",
    "r esults": "Results",
    "c onclusion": "Conclusion",
    "c onclusions": "Conclusion",
    "d atasets": "Datasets",
    "r elated work": "Related Work",
}

def normalize_ws(text: str) -> str:
    """规范换行和水平空白，同时去除文本首尾空白。"""
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def repair_heading(raw: str) -> str:
    """清理标题符号并修复已知 PDF OCR 标题错误。"""
    s = normalize_ws(raw).strip(" :-–—")
    low = s.casefold()
    if low in KNOWN_OCR_REPAIRS:
        return KNOWN_OCR_REPAIRS[low]
    return s

def canonical_section(raw: str) -> Optional[str]:
    """将原始标题映射到预定义的科研论文标准章节。"""
    # Repair known OCR glitches before treating a leading Roman numeral as numbering.
    s = repair_heading(raw)

    # If it is not an OCR-glitch label, strip numeric/Roman section numbering.
    if s.casefold() == raw.strip().casefold():
        s = NUMBERING_RE.sub("", s, count=1).strip()

    s = repair_heading(s)
    key = s.casefold()

    for canonical, pattern in SECTION_PATTERNS:
        if pattern.fullmatch(s):
            return canonical

    # Conservative mapping for paper-specific but clearly methodological/evaluation headings.
    if any(k in key for k in METHOD_HINTS):
        return "Method"
    if any(k in key for k in EXPERIMENT_HINTS):
        return "Experiments"

    return None

def detect_heading(line: str) -> Optional[str]:
    """保守判断单行文本是否为可识别的章节标题。"""
    s = normalize_ws(line)
    if not s or len(s) > 90:
        return None
    if s.endswith((".", ",", ";", ":", "?", "!")):
        return None
    if len(s.split()) > 12:
        return None
    return canonical_section(s)

def split_sections(text: str) -> List[Tuple[str, str]]:
    """按识别出的标题把论文文本拆分为章节及其正文。"""
    lines = text.splitlines()
    current = "Unknown"
    body: List[str] = []
    out: List[Tuple[str, str]] = []

    for line in lines:
        heading = detect_heading(line)
        if heading is not None:
            if body:
                section_text = "\n".join(body).strip()
                if section_text:
                    out.append((current, section_text))
            current = heading
            body = []
        else:
            body.append(line)

    if body:
        section_text = "\n".join(body).strip()
        if section_text:
            out.append((current, section_text))

    return out or [("Unknown", text)]

def sentence_split(text: str) -> List[str]:
    """在常见中英文句末标点后切分规范化文本。"""
    text = normalize_ws(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?。！？])\s+(?=[A-Z0-9(\[])", text)
    return [p.strip() for p in parts if p.strip()]

def split_long_unit(text: str, max_chars: int) -> List[str]:
    """Last-resort split at word boundaries; never cut a word in half."""
    text = normalize_ws(text)
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    out, buf = [], []
    n = 0
    for word in words:
        add = len(word) + (1 if buf else 0)
        if buf and n + add > max_chars:
            out.append(" ".join(buf))
            buf = [word]
            n = len(word)
        else:
            buf.append(word)
            n += add
    if buf:
        out.append(" ".join(buf))
    return out

def sentence_tail(text: str, target_chars: int) -> str:
    """Overlap with whole trailing sentences instead of raw characters."""
    if target_chars <= 0:
        return ""
    sents = sentence_split(text)
    if not sents:
        return ""
    chosen = []
    total = 0
    for sent in reversed(sents):
        extra = len(sent) + (1 if chosen else 0)
        if chosen and total + extra > target_chars:
            break
        chosen.append(sent)
        total += extra
        if total >= target_chars:
            break
    return " ".join(reversed(chosen))

def chunk_section(
    section_text: str,
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> List[str]:
    """优先按段落和句子边界将单个章节切成带语义重叠的文本块。"""
    paras = [
        normalize_ws(p)
        for p in re.split(r"\n\s*\n+", section_text)
        if normalize_ws(p)
    ]
    if not paras:
        paras = [normalize_ws(section_text)] if normalize_ws(section_text) else []

    chunks: List[str] = []
    buf = ""

    def flush():
        """保存当前缓冲区，并以完整尾句初始化下一个重叠缓冲区。"""
        nonlocal buf
        old = buf.strip()
        if len(old) >= 80:
            chunks.append(old)
        buf = sentence_tail(old, overlap_chars)

    for para in paras:
        if len(para) <= max_chars:
            candidate = (buf + "\n\n" + para).strip() if buf else para
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                flush()
                candidate = (buf + "\n\n" + para).strip() if buf else para
                if len(candidate) <= max_chars:
                    buf = candidate
                else:
                    for piece in split_long_unit(para, max_chars):
                        candidate = (buf + " " + piece).strip() if buf else piece
                        if len(candidate) <= max_chars:
                            buf = candidate
                        else:
                            flush()
                            buf = piece
            continue

        units: List[str] = []
        for sent in sentence_split(para):
            units.extend(split_long_unit(sent, max_chars))

        for unit in units:
            candidate = (buf + " " + unit).strip() if buf else unit
            if len(candidate) <= max_chars:
                buf = candidate
            else:
                flush()
                candidate = (buf + " " + unit).strip() if buf else unit
                buf = candidate if len(candidate) <= max_chars else unit

    if len(buf.strip()) >= 80:
        chunks.append(buf.strip())

    return chunks

def chunk_paper(
    paper: Dict,
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> List[Dict]:
    """为单篇论文生成带章节、论文和稳定 chunk ID 的结构化文本块。"""
    pid = paper.get("paper_id") or paper.get("id") or "unknown"
    title = paper.get("title", "")
    year = paper.get("year", "")
    text = paper.get("text", "")

    out: List[Dict] = []
    idx = 0
    for section, body in split_sections(text):
        for chunk in chunk_section(body, max_chars=max_chars, overlap_chars=overlap_chars):
            out.append({
                "paper_id": pid,
                "title": title,
                "year": year,
                "section": section,
                "chunk_id": f"{pid}_s{idx:04d}",
                "text": chunk,
            })
            idx += 1
    return out

def build_chunks(
    papers_path: str,
    output_path: str,
    max_chars: int = 1800,
    overlap_chars: int = 180,
) -> int:
    """读取论文 JSONL、生成结构化分块并返回输出记录数。"""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(papers_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:
        for line in fin:
            if not line.strip():
                continue
            paper = json.loads(line)
            for chunk in chunk_paper(
                paper,
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            ):
                fout.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
    return count
