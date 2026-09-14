import argparse,json
p=argparse.ArgumentParser();p.add_argument('--papers',default='data/processed/papers.jsonl');args=p.parse_args()
for x in open(args.papers,encoding='utf-8'):
    if x.strip():
        ppr=json.loads(x);print(ppr.get('paper_id'), '\t', ppr.get('title',''))
