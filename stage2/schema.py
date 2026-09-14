from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional

@dataclass
class Evidence:
    evidence_id: str
    source_type: str  # vector | triple | path
    content: str
    paper_id: str = ""
    title: str = ""
    section: str = ""
    retrieval_score: float = 0.0
    rerank_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    paper_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.paper_ids and self.paper_id:
            self.paper_ids = [self.paper_id]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["metadata"] = d["metadata"] or {}
        return d

@dataclass
class QAItem:
    qid: str
    question: str
    qtype: str
    gold_answer: str
    gold_papers: List[str]
