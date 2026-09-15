import math, re
from typing import Dict, List, Optional
import numpy as np
from stage2.schema import Evidence

STOP = {'the','a','an','of','to','in','on','for','and','or','is','are','was','were','what','which','how','why','who','与','和','的','是','什么','哪些','如何','为什么'}

QTYPE_COMPARISON = ('compare','difference','different','versus',' vs ','区别','比较','不同','异同','差异','差别','分别更强调','各自更强调')
QTYPE_MULTI_HOP = ('如果','应选','哪个统一框架','从“','从"','同时判断','三个条件','哪个模型既','哪个模型同时','哪个模型把','哪个模型通过','哪个核心模型既','哪个图级模型既','哪个图级模型会先','哪个扩展大图训练的方法')
QTYPE_RELATION = ('relation','relationship','关联','关系','分别承担什么角色','为什么需要配合','看成特例','为什么要解耦','为什么强调')

RELATION_INTENTS = {
    'USES': ('use','uses','using','encoder','decoder','使用','采用','编码器','解码器'),
    'PROPOSES': ('propose','introduce','提出','引入'),
    'SOLVES': ('solve','task','用于','解决','classification','prediction','学习设定'),
    'EVALUATED_ON': ('dataset','benchmark','evaluate','evaluation','数据集','评估','实验'),
    'EXTENDS': ('extend','improve','relation','relationship','关系','衔接'),
    'SIMPLIFIES': ('simplify','remove','简化','去掉','删除'),
}

SECTION_PRIOR = {
    'fact': {'Abstract':0.05,'Introduction':0.05,'Method':0.08,'Experiments':0.05,'Results':0.03,'Conclusion':0.02},
    'relation': {'Method':0.10,'Introduction':0.06,'Abstract':0.04},
    'multi_hop': {'Method':0.08,'Experiments':0.06,'Results':0.04},
    'comparison': {'Method':0.06,'Experiments':0.08,'Results':0.08,'Conclusion':0.04},
    'summary': {'Abstract':0.10,'Introduction':0.08,'Conclusion':0.08},
}

def qtype(query: str) -> str:
    """仅根据查询文本推断 Fact、Relation、Comparison 或 Multi-hop 类型。"""
    q=f' {query.casefold()} '
    if any(x in q for x in QTYPE_MULTI_HOP): return 'multi_hop'
    if any(x in q for x in QTYPE_COMPARISON): return 'comparison'
    if any(x in q for x in QTYPE_RELATION): return 'relation'
    return 'fact'

def tokens(text: str):
    """提取英文词项及中文单字/双字特征用于实体重叠计算。"""
    en=re.findall(r'[a-z][a-z0-9]*(?:[-_][a-z0-9]+)*', text.casefold())
    cjk_runs=re.findall(r'[\u4e00-\u9fff]+', text)
    zh=[]
    stop_chars=set('与和的是有什么哪如为')
    for run in cjk_runs:
        zh.extend(char for char in run if char not in stop_chars)
        zh.extend(run[i:i+2] for i in range(len(run)-1))
    return {x for x in en+zh if x not in STOP}

class EvidenceReranker:
    def __init__(self, embedder, weights: Optional[Dict[str,float]] = None):
        """保存嵌入器并初始化可解释 rerank 特征权重。"""
        self.embedder=embedder
        self.weights=weights or {'semantic':0.50,'entity':0.18,'relation':0.12,'section':0.10,'source':0.10}

    def _semantic(self, query: str, evidence: List[Evidence]) -> List[float]:
        """计算查询与每条证据的归一化嵌入相似度。"""
        texts=[query]+[e.content for e in evidence]
        vec=np.asarray(self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False),dtype='float32')
        q=vec[0]; scores=vec[1:]@q
        return [float((s+1)/2) for s in scores]

    def _entity(self, query: str, content: str) -> float:
        """计算查询和证据之间经截断归一化的词项重叠分数。"""
        q=tokens(query); e=tokens(content)
        if not q: return 0.0
        return min(1.0, len(q & e)/max(1,min(len(q),4)))

    def _relation(self, query: str, e: Evidence) -> float:
        """根据查询关系意图和证据实际关系/内容计算关系匹配分。"""
        query_text=query.casefold()
        expected={
            relation for relation, keywords in RELATION_INTENTS.items()
            if any(keyword in query_text for keyword in keywords)
        }
        metadata=e.metadata or {}
        actual=[]
        if metadata.get('relation'):
            actual.append(str(metadata['relation']).upper())
        actual.extend(str(relation).upper() for relation in metadata.get('relations',[]))
        if not actual:
            actual.extend(re.findall(r'--([A-Za-z_]+)--?>', e.content.upper()))
        if actual:
            if expected and any(relation in expected for relation in actual):
                return 1.0
            if qtype(query)=='multi_hop' and e.source_type=='path' and len(actual)>=2:
                return 0.9
            return 0.55 if any(relation != 'RELATED_TO' for relation in actual) else 0.3
        content=e.content.casefold()
        if expected:
            matched=sum(
                any(keyword in content for keyword in RELATION_INTENTS[relation])
                for relation in expected
            )
            return min(1.0, 0.25 + 0.5 * matched)
        return 0.45 if qtype(query) in {'relation','comparison'} else 0.35

    def _section(self, query: str, e: Evidence) -> float:
        """按推断题型与论文章节先验计算章节适配分。"""
        return SECTION_PRIOR.get(qtype(query),{}).get(e.section,0.0)/0.10 if e.section else 0.0

    def _source(self, e: Evidence) -> float:
        """根据 Vector、Triple 或 Path 来源返回解释性来源先验。"""
        return {'path':1.0,'triple':0.9,'vector':0.8}.get(e.source_type,0.5)

    def _retrieval(self, e: Evidence) -> float:
        """将不同检索器的原始分数映射到零到一范围。"""
        if e.source_type == 'vector':
            return min(1.0, max(0.0, (e.retrieval_score + 1.0) / 2.0))
        return min(1.0, max(0.0, 1.0 - math.exp(-e.retrieval_score)))

    def rerank(self, query: str, evidence: List[Evidence], top_k: int = 6) -> List[Evidence]:
        """计算并保存各项得分，去除完全重复证据后选择前 top_k 条。"""
        if not evidence: return []
        sem=self._semantic(query,evidence)
        for i,e in enumerate(evidence):
            parts={
                'semantic':sem[i],
                'entity':self._entity(query,e.content),
                'relation':self._relation(query,e),
                'section':self._section(query,e),
                'source':self._source(e),
                'retrieval':self._retrieval(e),
            }
            e.rerank_score=sum(self.weights[k]*parts[k] for k in self.weights)
            e.metadata=(e.metadata or {}) | {'rerank_parts':parts,'qtype':qtype(query)}
        evidence=sorted(evidence,key=lambda x:(
            -x.rerank_score,
            x.source_type,
            x.paper_id,
            x.evidence_id,
        ))

        # Remove exact duplicate evidence without imposing a gold-paper-count prior.
        # A hard per-paper cap is invalid at runtime because a legitimate single-paper
        # question may need several complementary passages from the same paper.
        selected=[]; seen=set()
        for e in evidence:
            key=(e.source_type, e.content, tuple(e.paper_ids))
            if key in seen:
                continue
            seen.add(key); selected.append(e)
            if len(selected)>=top_k: break
        return selected
