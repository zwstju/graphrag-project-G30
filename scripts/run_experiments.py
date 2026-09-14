import argparse,csv,json,os,time
from pathlib import Path
from stage2.pipeline import METHOD_LABELS, Stage2System
from stage2.eval.metrics import token_f1,retrieval_recall,evidence_precision
from stage2.eval.judge import llm_correctness

p=argparse.ArgumentParser()
p.add_argument('--config',default='configs/default.json')
p.add_argument('--benchmark',default='benchmark/gnn_qa_seed.jsonl')
p.add_argument('--methods',nargs='+',default=['vector','graph','hybrid','rerank','full'])
p.add_argument('--limit',type=int,default=0)
p.add_argument('--llm-judge',action='store_true')
p.add_argument('--verify-full',action='store_true')
p.add_argument('--output',default='results/ablation.csv')
args=p.parse_args()

items=[json.loads(x) for x in open(args.benchmark,encoding='utf-8') if x.strip()]
if args.limit: items=items[:args.limit]
sys=Stage2System(args.config)
Path(args.output).parent.mkdir(parents=True,exist_ok=True)
rows=[]
for method in args.methods:
    for i,it in enumerate(items,1):
        print(f'[{method}] {i}/{len(items)} {it["qid"]}: {it["question"]}')
        try:
            out=sys.answer(it['question'],method=method,verify=(args.verify_full and method=='full'))
            row={
                'qid':it['qid'],'qtype':it['qtype'],'method':method,
                'condition_label':METHOD_LABELS[method],
                'retrieval_metric_scope':'paper-level-proxy',
                'retrieval_recall':retrieval_recall(out['evidence'],it['gold_papers']),
                'evidence_precision':evidence_precision(out['evidence'],it['gold_papers']),
                'token_f1':token_f1(out['answer'],it['gold_answer']),
                'citation_validity':out['citation_check'].get('citation_validity',0),
                'retrieval_latency_s':out['retrieval_latency_s'],'generation_latency_s':out['generation_latency_s'],
                'answer':out['answer'],'error':''
            }
            if args.llm_judge:
                score,reason=llm_correctness(it['question'],it['gold_answer'],out['answer'])
                row['llm_correctness']=score; row['judge_reason']=reason
            cv=out['citation_check'].get('claim_verification')
            if isinstance(cv,dict): row['claim_support_rate']=cv.get('support_rate','')
        except Exception as e:
            row={
                'qid':it['qid'],'qtype':it['qtype'],'method':method,
                'condition_label':METHOD_LABELS.get(method,method),
                'retrieval_metric_scope':'paper-level-proxy','error':repr(e)
            }
        rows.append(row)
        with open(args.output,'w',newline='',encoding='utf-8-sig') as f:
            keys=sorted({k for r in rows for k in r.keys()})
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
print('saved',args.output)
