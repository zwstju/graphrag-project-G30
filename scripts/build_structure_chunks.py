import argparse, json
from stage2.data.structure_chunker import build_chunks

p = argparse.ArgumentParser()
p.add_argument('--config', default='configs/default.json')
args = p.parse_args()
conf = json.load(open(args.config, encoding='utf-8'))
n = build_chunks(conf['papers_path'], conf['chunks_path'], conf['chunk_max_chars'], conf['chunk_overlap_chars'])
print(f'Wrote {n} structure-aware chunks -> {conf["chunks_path"]}')
