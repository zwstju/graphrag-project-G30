import json
import re
from typing import Dict, List

import networkx as nx

from stage2.schema import Evidence

RELATION_KEYWORDS = {
    'uses': ['use','uses','using','encoder','based on','采用','使用','基于'],
    'proposes': ['propose','introduce','提出','引入'],
    'solves': ['solve','task','用于','解决','prediction','classification'],
    'evaluated_on': ['dataset','benchmark','evaluate','实验','数据集','评估'],
    'extends': ['extend','improve','改进','扩展'],
    'simplifies': ['simplify','remove','简化','去除'],
}

QA_NOISE_RELATIONS = {'WRITTEN_BY', 'CITES'}
RELATION_PRIOR = {
    'USES': 1.0,
    'PROPOSES': 0.95,
    'SOLVES': 0.9,
    'EVALUATED_ON': 0.85,
    'EXTENDS': 0.8,
    'SIMPLIFIES': 0.8,
    'RELATED_TO': 0.4,
}

def norm(s: str) -> str:
    return re.sub(r'[^a-z0-9\u4e00-\u9fff]+', '', (s or '').casefold())


def relation_key(s: str) -> str:
    return re.sub(r'[^A-Z0-9]+', '_', (s or '').upper()).strip('_')


def latin_tokens(s: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', (s or '').casefold())


def entity_occurs(query: str, entity: str) -> bool:
    """Match Latin entities on token boundaries and CJK entities literally."""
    entity_latin = latin_tokens(entity)
    query_latin = latin_tokens(query)
    if entity_latin:
        width = len(entity_latin)
        if any(query_latin[i:i + width] == entity_latin for i in range(len(query_latin) - width + 1)):
            return True
    entity_cjk = ''.join(re.findall(r'[\u4e00-\u9fff]+', entity))
    query_cjk = ''.join(re.findall(r'[\u4e00-\u9fff]+', query))
    return len(entity_cjk) >= 2 and entity_cjk in query_cjk

class GraphRetriever:
    def __init__(self, triples_path: str, max_hops: int = 3):
        self.g = nx.MultiDiGraph()
        self.qa_graph = nx.MultiDiGraph()
        self.triples = []
        self.max_hops = max_hops
        self.entity_by_norm: Dict[str, str] = {}
        self._load(triples_path)

    def _load(self, path: str):
        with open(path, encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                t = json.loads(line)
                s = t.get('source') or t.get('head') or ''
                r = t.get('relation') or t.get('rel') or 'RELATED_TO'
                o = t.get('target') or t.get('tail') or ''
                if not s or not o: continue
                pid = t.get('paper_id','')
                title = t.get('title','')
                self.triples.append((s,r,o,pid,title,t))
                self.g.add_edge(s,o,key=len(self.triples)-1,relation=r,paper_id=pid,title=title,raw=t)
                if relation_key(r) not in QA_NOISE_RELATIONS:
                    self.qa_graph.add_edge(
                        s, o, key=len(self.triples)-1, relation=r,
                        paper_id=pid, title=title, raw=t,
                    )
                self.entity_by_norm.setdefault(norm(s), s)
                self.entity_by_norm.setdefault(norm(o), o)

    def find_seeds(self, query: str, limit: int = 6) -> List[str]:
        hits = []
        for nk, raw in self.entity_by_norm.items():
            if len(nk) < 2: continue
            if entity_occurs(query, raw):
                hits.append((len(nk), nk, raw))
        hits.sort(key=lambda item: (-item[0], item[1], item[2].casefold()))
        chosen=[]
        for _, _, e in hits:
            ne=norm(e)
            # suppress short substring aliases if a longer entity already matched
            if any(ne in norm(x) and len(ne) < len(norm(x)) for x in chosen):
                continue
            chosen.append(e)
            if len(chosen)>=limit: break
        return chosen

    def _relation_bonus(self, query: str, relation: str) -> float:
        q = query.casefold(); r = norm(relation)
        bonus=0.0
        for family, kws in RELATION_KEYWORDS.items():
            if norm(family) in r and any(k.casefold() in q for k in kws):
                bonus += 0.3
        return bonus

    def _edge_score(self, query: str, relation: str) -> float:
        key = relation_key(relation)
        return RELATION_PRIOR.get(key, 0.5) + self._relation_bonus(query, relation)

    def _choose_edge(self, query: str, u: str, v: str):
        candidates = []
        for reverse, start, end in ((False, u, v), (True, v, u)):
            data = self.qa_graph.get_edge_data(start, end) or {}
            for key, edge in data.items():
                relation = edge.get('relation', 'RELATED_TO')
                candidates.append((
                    -self._edge_score(query, relation),
                    norm(relation),
                    edge.get('paper_id', ''),
                    reverse,
                    str(key),
                    edge,
                ))
        if not candidates:
            return None
        return sorted(candidates)[0]

    def _shortest_path(self, graph, source: str, target: str) -> List[str]:
        paths = nx.all_shortest_paths(graph, source, target)
        return min(paths, key=lambda path: tuple((norm(node), node.casefold()) for node in path))

    def search(self, query: str, top_k: int = 12) -> List[Evidence]:
        if top_k <= 0:
            return []
        seeds = self.find_seeds(query)
        scored=[]
        for i,(s,r,o,pid,title,raw) in enumerate(self.triples):
            if relation_key(r) in QA_NOISE_RELATIONS:
                continue
            score=0.0
            if s in seeds: score+=1.0
            if o in seeds: score+=1.0
            score += self._relation_bonus(query,r)
            if score>0:
                scored.append((score,i,s,r,o,pid,title,raw))
        scored.sort(key=lambda x:(-x[0], norm(x[3]), norm(x[2]), norm(x[4]), x[5], x[1]))
        out=[]
        for rank,(score,i,s,r,o,pid,title,raw) in enumerate(scored):
            out.append(Evidence(
                evidence_id=f'G{rank+1}', source_type='triple',
                content=f'{s} --{r}--> {o}', paper_id=pid, title=title,
                retrieval_score=float(score),
                metadata={'source':s,'relation':r,'target':o,'raw':raw,'paper_ids':[pid] if pid else []},
                paper_ids=[pid] if pid else [],
            ))

        # Explicit paths between matched entities. This is evidence, not unrestricted reasoning.
        path_rank=1
        if len(seeds)>=2:
            ug = self.qa_graph.to_undirected()
            for a_i in range(len(seeds)):
                for b_i in range(a_i+1,len(seeds)):
                    a,b=seeds[a_i],seeds[b_i]
                    try:
                        nodes=self._shortest_path(ug,a,b)
                    except (nx.NetworkXNoPath,nx.NodeNotFound):
                        continue
                    hops=len(nodes)-1
                    if hops<1 or hops>self.max_hops: continue
                    rels=[]; pids=[]; edge_scores=[]
                    for u,v in zip(nodes,nodes[1:]):
                        choice = self._choose_edge(query, u, v)
                        if choice is None:
                            continue
                        neg_score, _, _, reverse, _, edge = choice
                        rel = edge.get('relation','RELATED_TO')
                        rels.append((u,rel,v,reverse))
                        edge_scores.append(-neg_score)
                        if edge.get('paper_id'): pids.append(edge.get('paper_id'))
                    if not rels: continue
                    text=' ; '.join([f'{u} --{r}--> {v}' if not rev else f'{u} <--{r}-- {v}' for u,r,v,rev in rels])
                    unique_pids=sorted(set(pids))
                    out.append(Evidence(
                        evidence_id=f'P{path_rank}', source_type='path', content=text,
                        paper_id=unique_pids[0] if unique_pids else '',
                        paper_ids=unique_pids,
                        retrieval_score=0.8 + 1.5/(1+hops) + (sum(edge_scores)/len(edge_scores))*0.2,
                        metadata={
                            'nodes':nodes,'hops':hops,'paper_ids':unique_pids,
                            'relations':[relation for _,relation,_,_ in rels],
                        }
                    ))
                    path_rank+=1
        out.sort(key=lambda e: (
            -e.retrieval_score,
            0 if e.source_type == 'triple' else 1,
            norm(e.content),
            tuple(e.paper_ids),
        ))
        selected=[]; per_paper={}
        for evidence in out:
            key=tuple(evidence.paper_ids) or ('',)
            if per_paper.get(key,0)>=2:
                continue
            selected.append(evidence)
            per_paper[key]=per_paper.get(key,0)+1
            if len(selected)>=top_k:
                return selected
        for evidence in out:
            if evidence not in selected:
                selected.append(evidence)
                if len(selected)>=top_k:
                    break
        return selected
