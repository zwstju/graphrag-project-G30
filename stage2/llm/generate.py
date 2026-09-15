from typing import List, Optional
from stage2.schema import Evidence
from stage2.llm.client import LLMClient

def build_prompt(question: str, evidence: List[Evidence]) -> str:
    """把问题和统一 Evidence 列表组织为带论文引用约束的生成提示词。"""
    blocks=[]
    for i,e in enumerate(evidence,1):
        pid=','.join(e.paper_ids) if e.paper_ids else (e.paper_id or 'unknown')
        sec=f' / {e.section}' if e.section else ''
        blocks.append(f'[E{i}] source={e.source_type}; paper={pid}{sec}\n{e.content}')
    return f'''你是科研论文问答助手。只能依据给定 Evidence 回答；若证据不足，明确说证据不足，不要补充外部事实。
每个关键事实后必须引用支持它的 paper_id，格式为 [paper_id]。不要伪造引用。

Question:
{question}

Evidence:
{chr(10).join(blocks)}

请给出简洁、技术准确、可追溯的回答。'''

def answer_question(question: str, evidence: List[Evidence], client: Optional[LLMClient] = None) -> str:
    """调用指定或默认 LLM，根据给定证据生成可追溯回答。"""
    client=client or LLMClient()
    return client.chat([
        {'role':'system','content':'Answer scientific questions only from supplied evidence and cite paper_id.'},
        {'role':'user','content':build_prompt(question,evidence)}
    ],temperature=0.0,max_tokens=1400)
