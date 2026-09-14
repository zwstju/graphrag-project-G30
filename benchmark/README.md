# Benchmark status — read before reporting numbers

This directory now has **two different scales** and they must not be conflated.

## Paper manifest

- `gnn_papers_expanded.jsonl`: **50 target GNN papers**, spanning convolution, sampling/scalability, pooling, self-supervision, heterophily, recommendation and graph transformers.
- `gnn_paper_titles.txt`: the same 50 titles for simple title-based fetch workflows.

A 50-paper manifest is **not** the same as a 50-paper downloaded corpus. The real corpus size is the number of successfully fetched/parsed rows in `data/processed/papers_stage2_expanded.jsonl`.

## QA benchmark

- `gnn_qa_seed.jsonl`: original **48 candidate QA** over the core 12-paper corpus.
- `gnn_qa_core80.jsonl`: **80 candidate QA** over the same 12-paper corpus, balanced as 20 fact / 20 relation / 20 comparison / 20 multi-hop. **This is the recommended tomorrow-morning benchmark if your graph has only been built for the original 12 papers.**
- `gnn_qa_expanded.jsonl`: **120 candidate QA**, balanced as 30 fact / 30 relation / 30 comparison / 30 multi-hop, referencing 34 distinct paper IDs and intended for the expanded corpus.

All 120 QA are deliberately marked `review_required=true`. They are research-test candidates, not automatically trustworthy gold labels.

## Promotion protocol

1. Make sure referenced `gold_papers` exist in the real corpus.
2. Build structure-aware chunks.
3. Run evidence-grounded audit:

```bash
python scripts/audit_benchmark_llm.py \
  --benchmark benchmark/gnn_qa_expanded.jsonl \
  --output results/benchmark_audit.csv
```

4. Filter high-confidence supported items:

```bash
python scripts/promote_verified_benchmark.py \
  --benchmark benchmark/gnn_qa_expanded.jsonl \
  --audit results/benchmark_audit.csv \
  --output benchmark/gnn_qa_verified.jsonl
```

5. **Manually inspect a stratified sample** (at minimum 5 per question type) against the source paper before calling the set verified.
6. Only then create dev/test splits with `build_benchmark_splits.py`.

## What to say in the presentation

Good:

> We prepared a 50-paper target corpus and 120 stratified QA candidates. For reported quantitative results, we only use the subset whose source papers were successfully ingested and whose answers passed evidence-grounded audit; a stratified sample was manually checked.

Bad:

> We have 50 papers and 120 verified QA.

unless the downloads and verification were actually completed.
