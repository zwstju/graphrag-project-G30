"""Expand the scientific corpus from a reviewed paper manifest.

Uses arXiv title search, preserves the manifest paper_id, downloads PDFs and extracts text.
The manifest is a TARGET list, not proof that all papers were downloaded. Report only the
actual count printed/written by this script.
"""
import argparse, json, re, time
from pathlib import Path
import arxiv
import fitz

p=argparse.ArgumentParser()
p.add_argument('--manifest',default='benchmark/gnn_papers_expanded.jsonl')
p.add_argument('--titles',default='benchmark/gnn_paper_titles.txt',help='fallback if manifest missing')
p.add_argument('--out',default='data/processed/papers_stage2_expanded.jsonl')
p.add_argument('--pdf-dir',default='data/raw_stage2')
p.add_argument('--limit',type=int,default=0)
p.add_argument('--skip-existing',action='store_true')
p.add_argument('--sleep',type=float,default=0.0)
args=p.parse_args()

def slug(s): return re.sub(r'[^a-z0-9]+','_',s.casefold()).strip('_')[:55]
def norm(s): return re.sub(r'[^a-z0-9]+',' ',s.casefold()).strip()

manifest_path=Path(args.manifest)
if manifest_path.exists():
    items=[json.loads(x) for x in manifest_path.read_text(encoding='utf-8').splitlines() if x.strip()]
else:
    titles=[x.strip() for x in open(args.titles,encoding='utf-8') if x.strip()]
    items=[{'paper_id':slug(t),'title':t,'year':''} for t in titles]
if args.limit: items=items[:args.limit]

Path(args.pdf_dir).mkdir(parents=True,exist_ok=True)
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
existing={}
if args.skip_existing and Path(args.out).exists():
    for line in open(args.out,encoding='utf-8'):
        if line.strip():
            r=json.loads(line);existing[r.get('paper_id')]=r

client=arxiv.Client(page_size=5,delay_seconds=3,num_retries=2)
rows=[]; failures=[]
for i,it in enumerate(items,1):
    pid=it['paper_id']; title=it['title']
    if pid in existing:
        rows.append(existing[pid]);print(f'[{i}/{len(items)}] SKIP {pid}');continue
    print(f'[{i}/{len(items)}] {pid}: {title}')
    try:
        results=list(client.results(arxiv.Search(query=f'ti:"{title}"',max_results=5,sort_by=arxiv.SortCriterion.Relevance)))
    except Exception as e:
        failures.append({'paper_id':pid,'title':title,'reason':f'search:{e!r}'});print('  SEARCH ERROR',repr(e));continue
    if not results:
        failures.append({'paper_id':pid,'title':title,'reason':'not_found'});print('  NOT FOUND');continue
    # Pick title with highest token overlap rather than blindly taking result[0].
    target=set(norm(title).split())
    def overlap(r):
        cand=set(norm(r.title).split());return len(target & cand)/max(1,len(target | cand))
    r=max(results,key=overlap); score=overlap(r)
    pdf=Path(args.pdf_dir)/f'{pid}.pdf'
    try:
        r.download_pdf(dirpath=args.pdf_dir, filename=pdf.name)
        doc=fitz.open(pdf)
        text='\n'.join(page.get_text('text') for page in doc)
        row={
            'paper_id':pid,'title':r.title,'manifest_title':title,
            'authors':[a.name for a in r.authors],
            'year':r.published.year if r.published else it.get('year',''),
            'arxiv_id':r.get_short_id(),'title_match_jaccard':round(score,4),
            'family':it.get('family',''),'text':text,
            'source':'arxiv_title_search'
        }
        rows.append(row);print(' ',r.get_short_id(),'match=',round(score,3),'chars=',len(text))
    except Exception as e:
        failures.append({'paper_id':pid,'title':title,'reason':f'download_or_parse:{e!r}'});print('  ERROR',repr(e))
    if args.sleep: time.sleep(args.sleep)

with open(args.out,'w',encoding='utf-8') as f:
    for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
fail_path=Path(args.out).with_name('papers_stage2_failures.jsonl')
with open(fail_path,'w',encoding='utf-8') as f:
    for x in failures:f.write(json.dumps(x,ensure_ascii=False)+'\n')
print(f'wrote {len(rows)}/{len(items)} papers -> {args.out}; failures={len(failures)} -> {fail_path}')
