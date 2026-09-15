import re
from typing import Dict, List

def norm(s):
    """将文本转换为便于词级指标比较的规范形式。"""
    return re.sub(r'\s+',' ',re.sub(r'[^a-z0-9\u4e00-\u9fff ]+',' ',(s or '').casefold())).strip()

def token_f1(pred: str, gold: str) -> float:
    """计算预测答案和候选答案之间的词项 F1。"""
    p=norm(pred).split(); g=norm(gold).split()
    if not p or not g: return 0.0
    from collections import Counter
    common=sum((Counter(p)&Counter(g)).values())
    if not common: return 0.0
    pr=common/len(p); rc=common/len(g)
    return 2*pr*rc/(pr+rc)

def retrieval_recall(evidence: List[Dict], gold_papers: List[str]) -> float:
    """Paper-level proxy recall; this is not passage-level evidence recall."""
    gp=set(gold_papers); got=set()
    for e in evidence:
        got.update(e.get('paper_ids') or [])
        got.update((e.get('metadata') or {}).get('paper_ids') or [])
        if e.get('paper_id'): got.add(e['paper_id'])
    return len(gp & got)/len(gp) if gp else 0.0

def evidence_precision(evidence: List[Dict], gold_papers: List[str]) -> float:
    """Paper-level proxy precision; this is not passage-level evidence precision."""
    gp=set(gold_papers)
    usable=[]
    for e in evidence:
        papers=set(e.get('paper_ids') or [])
        papers.update((e.get('metadata') or {}).get('paper_ids') or [])
        if e.get('paper_id'): papers.add(e['paper_id'])
        if papers: usable.append(papers)
    if not usable: return 0.0
    return sum(1 for papers in usable if papers & gp)/len(usable)
