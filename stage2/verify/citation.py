import json, re
from typing import List, Dict, Optional
from stage2.schema import Evidence
from stage2.llm.client import LLMClient

def cited_papers(answer: str):
    return re.findall(r'\[([A-Za-z0-9_.-]+)\]', answer)

def deterministic_citation_check(answer: str, evidence: List[Evidence]) -> Dict:
    cited=cited_papers(answer)
    available=set()
    for e in evidence:
        available.update(e.paper_ids)
        if e.paper_id: available.add(e.paper_id)
    valid=[c for c in cited if c in available]
    return {
        'citations':cited,
        'valid_citations':valid,
        'citation_validity': len(valid)/len(cited) if cited else 0.0,
        'available_papers':sorted(available)
    }


def split_atomic_claims(answer: str) -> List[str]:
    """Deterministically segment an answer before asking the LLM for judgments."""
    cleaned=re.sub(r'\[([A-Za-z0-9_.-]+)\]','',answer or '')
    segments=re.split(r'(?:\n+|(?<=[。！？!?；;])\s*)',cleaned)
    claims=[]
    for segment in segments:
        claim=re.sub(r'^\s*(?:[-*•]|\d+[.)、])\s*','',segment).strip()
        claim=claim.strip('。！？!?；; ')
        if claim and len(claim)>=4:
            claims.append(claim)
    return claims


def verify_claims_llm(question: str, answer: str, evidence: List[Evidence], client: Optional[LLMClient] = None) -> Dict:
    client=client or LLMClient()
    ev='\n'.join([f'E{i+1} paper={e.paper_id}: {e.content[:1200]}' for i,e in enumerate(evidence)])
    claims=split_atomic_claims(answer)
    if not claims:
        raise ValueError('No deterministic claims could be extracted from the answer')
    claim_text='\n'.join(f'C{i+1}: {claim}' for i,claim in enumerate(claims))
    prompt=f'''你是严格的事实核验器。Claims 已经确定性切分。逐条判断是否被 Evidence 直接支持，不要改写、合并或新增 claim。
只返回 JSON，不要 Markdown：
{{"judgments":[{{"claim_id":"C1","supported":true,"evidence_ids":["E1"]}}]}}
必须恰好返回 C1 到 C{len(claims)} 各一次。supported=true 时 evidence_ids 必须非空且只能使用给定 Evidence ID。

Question: {question}
Answer: {answer}
Claims:\n{claim_text}
Evidence:\n{ev}'''
    raw=client.chat([{'role':'user','content':prompt}],temperature=0.0,max_tokens=1200)
    m=re.search(r'\{.*\}',raw,re.S)
    obj=json.loads(m.group(0) if m else raw)
    judgments=obj.get('judgments',[])
    by_id={item.get('claim_id'):item for item in judgments}
    expected={f'C{i+1}' for i in range(len(claims))}
    if set(by_id)!=expected or len(judgments)!=len(claims):
        raise ValueError(f'Claim judgment IDs mismatch: expected={sorted(expected)}, got={sorted(str(x) for x in by_id)}')
    valid_evidence_ids={f'E{i+1}' for i in range(len(evidence))}
    output=[]
    for index,claim in enumerate(claims,1):
        judgment=by_id[f'C{index}']
        evidence_ids=judgment.get('evidence_ids') or []
        if any(evidence_id not in valid_evidence_ids for evidence_id in evidence_ids):
            raise ValueError(f'Invalid evidence ID for C{index}: {evidence_ids}')
        if judgment.get('supported') and not evidence_ids:
            raise ValueError(f'Supported claim C{index} has no evidence IDs')
        output.append({
            'claim_id':f'C{index}','claim':claim,
            'supported':bool(judgment.get('supported')),
            'evidence_ids':evidence_ids,
        })
    sup=sum(1 for claim in output if claim['supported'])
    return {
        'claim_segmentation':'deterministic sentence/semicolon splitting',
        'claims':output,
        'support_rate':sup/len(output),
        'supported_claims':sup,
        'total_claims':len(output),
        'raw_verification_response':raw,
    }
