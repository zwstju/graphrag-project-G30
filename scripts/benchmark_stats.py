import argparse,json
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--benchmark',default='benchmark/gnn_qa_expanded.jsonl');args=p.parse_args()
items=[json.loads(x) for x in open(args.benchmark,encoding='utf-8') if x.strip()]
print('QA total:',len(items))
print('qtype:',dict(Counter(x.get('qtype') for x in items)))
print('difficulty:',dict(Counter(x.get('difficulty','') for x in items)))
print('review_status:',dict(Counter(x.get('review_status','') for x in items)))
papers=Counter()
for x in items:
    for pid in x.get('gold_papers',[]):papers[pid]+=1
print('gold-paper coverage:',len(papers))
print('top cited gold papers:')
for pid,n in papers.most_common(20):print(f'  {pid}: {n}')
