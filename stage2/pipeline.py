import json, os, time
from typing import Dict, List
from dotenv import load_dotenv
from stage2.retrieval.vector import VectorIndex
from stage2.retrieval.graph import GraphRetriever
from stage2.rerank.evidence_reranker import EvidenceReranker
from stage2.llm.generate import answer_question
from stage2.verify.citation import deterministic_citation_check, verify_claims_llm

load_dotenv()

METHOD_LABELS = {
    'vector': 'Stage2-Structured-Vector',
    'graph': 'Stage2-Graph',
    'hybrid': 'Stage2-Structured-Vector+Graph-Direct-Selection',
    'rerank': 'Stage2-Structured-Hybrid+Rerank',
    'full': 'Stage2-Rerank+PostHoc-Verification',
}

class Stage2System:
    def __init__(self, config_path='configs/default.json'):
        self.config=json.load(open(config_path,encoding='utf-8'))
        model=os.getenv('EMBED_MODEL',self.config['embedding_model'])
        device=self.config.get('embedding_device','auto')
        self.vector=VectorIndex(model,device=device).load(
            self.config['faiss_path'],
            self.config['vector_meta_path'],
            self.config.get('vector_manifest_path'),
            self.config.get('retrieval_chunks_path'),
        )
        self.graph=GraphRetriever(self.config['triples_path'],self.config.get('graph_max_hops',3))
        self.reranker=EvidenceReranker(self.vector.model,self.config.get('rerank_weights'))

    def retrieve(self, query: str, method: str='full'):
        # These are Stage 2 controlled conditions, not the original Stage 1
        # implementations: vector uses structured chunks and graph uses configured hops.
        vc=self.config.get('vector_candidates',12); gc=self.config.get('graph_candidates',12); fk=self.config.get('final_evidence_k',6)
        if method=='vector': return self.vector.search(query,fk)
        if method=='graph': return self.graph.search(query,fk)
        v=self.vector.search(query,vc); g=self.graph.search(query,gc)
        if method=='hybrid': return (v[:max(1,fk//2)] + g[:fk-max(1,fk//2)])[:fk]
        # "full" has the same retrieval as "rerank". Its verifier is post-hoc and
        # measures support/auditability; it does not improve answers by regeneration.
        if method in {'rerank','full'}: return self.reranker.rerank(query,v+g,fk)
        raise ValueError(f'unknown method={method}')

    def answer(self, query: str, method: str='full', verify: bool=False) -> Dict:
        t0=time.perf_counter(); ev=self.retrieve(query,method); t1=time.perf_counter()
        ans=answer_question(query,ev); t2=time.perf_counter()
        checks=deterministic_citation_check(ans,ev)
        if verify and method=='full':
            try: checks['claim_verification']=verify_claims_llm(query,ans,ev)
            except Exception as e: checks['claim_verification']={'error':str(e)}
        return {
            'question':query,'method':method,'answer':ans,'evidence':[e.to_dict() for e in ev],
            'citation_check':checks,'retrieval_latency_s':t1-t0,'generation_latency_s':t2-t1,'total_latency_s':t2-t0
        }
