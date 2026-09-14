import argparse,json
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('--benchmark',default='benchmark/gnn_qa_expanded.jsonl');p.add_argument('--papers',default='data/processed/papers.jsonl');args=p.parse_args()
items=[json.loads(x) for x in open(args.benchmark,encoding='utf-8') if x.strip()]
try: papers={json.loads(x).get('paper_id') for x in open(args.papers,encoding='utf-8') if x.strip()}
except FileNotFoundError: papers=set();print('WARNING: corpus file not found:',args.papers)
missing=[];needed=set()
for it in items:
    for pid in it.get('gold_papers',[]):
        needed.add(pid)
        if pid not in papers: missing.append((it['qid'],pid))
print('QA:',len(items),'types:',Counter(x['qtype'] for x in items))
print('benchmark gold-paper coverage:',len(needed))
print('corpus paper_ids:',len(papers))
print('gold papers present:',len(needed & papers),'/',len(needed))
if missing:
    print('WARNING: benchmark references not present in corpus:',len(missing))
    for x in missing[:50]: print(' ',x)
    if len(missing)>50: print(' ...',len(missing)-50,'more')
    print('Use the 12-paper seed benchmark now, or expand corpus before reporting the 120-QA benchmark.')
else: print('All benchmark paper IDs exist in corpus.')
print('NOTE: candidate benchmark items are model-authored. Evidence-audit and manually inspect before reporting as verified gold data.')
