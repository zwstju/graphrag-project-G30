import argparse,json,random
from collections import defaultdict
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--benchmark',default='benchmark/gnn_qa_verified.jsonl');p.add_argument('--out-dir',default='benchmark/splits');p.add_argument('--dev-ratio',type=float,default=.34);p.add_argument('--seed',type=int,default=42);args=p.parse_args()
items=[json.loads(x) for x in open(args.benchmark,encoding='utf-8') if x.strip()]
rng=random.Random(args.seed); groups=defaultdict(list)
for x in items:groups[x['qtype']].append(x)
dev=[];test=[]
for typ,xs in sorted(groups.items()):
    rng.shuffle(xs);n=round(len(xs)*args.dev_ratio);dev+=xs[:n];test+=xs[n:]
Path(args.out_dir).mkdir(parents=True,exist_ok=True)
for name,xs in [('dev',dev),('test',test)]:
    with open(Path(args.out_dir)/f'{name}.jsonl','w',encoding='utf-8') as f:
        for x in xs:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print('dev',len(dev),'test',len(test),'types_dev',{t:sum(x['qtype']==t for x in dev) for t in groups},'types_test',{t:sum(x['qtype']==t for x in test) for t in groups})
