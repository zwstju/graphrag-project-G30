# Stage 2: Evidence-Centric Scientific HybridRAG

## 0. Reviewer-first problem statement

**Do not claim novelty from “VectorRAG + GraphRAG”, “2-hop”, “semantic chunking”, or “better UI”.** Those are implementation choices, not a research contribution.

This stage asks a falsifiable question:

> In scientific-paper QA, vanilla HybridRAG often retrieves semantically similar but logically irrelevant context. Can structure-aware chunking + heterogeneous evidence reranking + citation verification improve answer correctness and evidence precision without an unacceptable latency increase?

The project should be presented as **Evidence-Centric Scientific HybridRAG**, not as a generic GraphRAG demo.

## 1. Minimum bar before claiming success

You must have all of the following before the presentation:

1. A reviewed benchmark with question type, gold answer, and gold paper provenance. The package now contains 120 **candidate** QA (30 fact / 30 relation / 30 comparison / 30 multi-hop), but only evidence-audited items may be reported as benchmark results.
2. Controlled baselines under the same corpus / question set / LLM / prompt.
3. A concrete reranking mechanism, not direct Vector+Graph concatenation.
4. At least one ablation table.
5. At least two success cases and two failure cases.
6. Retrieval latency and evidence-quality metrics, not only answer screenshots.
7. Every reported benchmark item must be evidence-grounded or explicitly marked unverified.

## 2. Method

```text
Scientific Query
      |
      v
Structure-aware corpus
(section -> paragraph -> fallback split)
      |
      +--------------------+
      |                    |
      v                    v
Vector candidates      Graph candidates
(BGE-M3 + FAISS)       (triples + <=3-hop paths)
      |                    |
      +---------+----------+
                v
        Unified Evidence Pool
                |
                v
       Scientific Evidence Reranker
       semantic + entity + relation
       + section prior + source prior
                |
                v
          Top-k Evidence
                |
                v
         Grounded LLM answer
                |
                v
      Citation / Claim Verification
```

The key contribution is **candidate evidence quality control**, not routing alone.

## 3. Included modules

- `stage2/data/structure_chunker.py`
  - Detects common scientific-paper sections.
  - Chunks inside section boundaries.
  - Preserves `section`, `paper_id`, `chunk_id` metadata.

- `stage2/retrieval/vector.py`
  - BGE-M3 embeddings.
  - normalized embedding + FAISS `IndexFlatIP`.
  - persisted index; runtime no longer re-embeds the whole corpus.

- `stage2/retrieval/graph.py`
  - loads existing `data/graph/triples.jsonl`.
  - lexical seed entity linking.
  - triple evidence + shortest path evidence up to configurable 3 hops.

- `stage2/rerank/evidence_reranker.py`
  - unified Vector / triple / path evidence reranking.
  - score = semantic + entity overlap + relation suitability + section prior + source prior.
  - greedy diversity selection prevents one paper/source dominating all evidence slots.

- `stage2/verify/citation.py`
  - deterministic citation validity check.
  - optional LLM claim-level support verification.

- `scripts/run_experiments.py`
  - runs Vector, Graph, Vanilla Hybrid, Hybrid+Rerank, Full.
  - records retrieval recall, evidence precision, answer metric, citation validity, latency.

- `benchmark/gnn_qa_seed.jsonl`
  - 48 seed QA over the original 12-paper corpus.
- `benchmark/gnn_qa_expanded.jsonl`
  - 120 candidate QA: 30 fact / 30 relation / 30 comparison / 30 multi-hop.
- `benchmark/gnn_papers_expanded.jsonl`
  - 50-paper target manifest spanning major GNN method families.
- **Every candidate QA is marked `review_required=true`. A target manifest is not a downloaded corpus. Do not report 50 papers / 120 verified QA unless ingestion and evidence audit actually succeeded.**

## 3.5 Expanded data layer

The v2 package now provides **two evaluation scales**:

```text
Immediate controlled benchmark (same existing graph corpus):
12 core papers
80 candidate QA
  20 fact / 20 relation / 20 comparison / 20 multi-hop

Expanded target benchmark:
50 target papers
120 candidate QA
  30 fact / 30 relation / 30 comparison / 30 multi-hop
```

The 80-QA core benchmark references exactly the original 12 paper IDs, so it can be used without rebuilding the graph for 50 papers. The 120-QA expanded benchmark references 34 distinct gold-paper IDs; the remaining manifest papers broaden the corpus and provide future coverage/distractors.

Use:

```bash
python scripts/benchmark_stats.py --benchmark benchmark/gnn_qa_core80.jsonl
python scripts/benchmark_stats.py --benchmark benchmark/gnn_qa_expanded.jsonl
python scripts/fetch_gnn_corpus.py --manifest benchmark/gnn_papers_expanded.jsonl
```

Then merge/replace the corpus path only after checking title matches and actual download counts. For tomorrow's report, distinguish **target manifest size**, **successfully ingested corpus size**, and **audited QA size**.

## 4. Required repository paths

This overlay assumes your existing repo already contains:

```text
data/processed/papers.jsonl
    paper_id, title, authors, year, text

data/graph/triples.jsonl
    source, relation, target, paper_id, ...
```

If your `paper_id` names differ from the seed benchmark, run:

```bash
python scripts/validate_benchmark.py
```

and update `gold_papers` before experiments.

## 5. Tonight: exact execution order

Activate the existing environment first. On fish:

```fish
source .venv/bin/activate.fish
pip install -r requirements.txt
cp .env.example .env
# Fill LLM_API_KEY / API base / model.
```

If BGE-M3 is already cached and HuggingFace network requests are unstable:

```fish
set -x HF_HUB_OFFLINE 1
set -x TRANSFORMERS_OFFLINE 1
```

### Recommended tomorrow-morning path

If your KG / graph index still covers only the original 12 papers, **do not mix it with a 50-paper vector index for the main ablation**. Use the controlled 12-paper / 80-QA candidate set first:

```bash
python scripts/validate_benchmark.py --benchmark benchmark/gnn_qa_core80.jsonl
python scripts/audit_benchmark_llm.py \
  --benchmark benchmark/gnn_qa_core80.jsonl \
  --output results/benchmark_audit_core80.csv
```

After audit, run experiments with `--benchmark benchmark/gnn_qa_core80.jsonl` or with the promoted audited subset. This keeps Vector / Graph / Hybrid on the **same corpus**, which is methodologically more important than inflating the paper count overnight.

### Step A — validate benchmark IDs

```bash
python scripts/validate_benchmark.py
```

Fix any missing `paper_id` mappings first.

### Step B — build structure-aware chunks

```bash
python scripts/build_structure_chunks.py
```

Inspect several lines manually:

```bash
head -n 3 data/stage2/structure_chunks.jsonl
```

Do not claim structure-aware chunking works until section metadata is plausible on GCN / VGAE / GAT.

### Step C — build persisted vector index

```bash
python scripts/build_vector_index.py
```

This fixes the Stage-1 engineering defect where every new process rebuilt embeddings.

### Step D — audit benchmark

Start with the original 48-item seed if only the core 12 papers are ingested:

```bash
python scripts/audit_benchmark_llm.py
```

If the expanded corpus has been ingested, audit the 120-item candidate set:

```bash
python scripts/audit_benchmark_llm.py \
  --benchmark benchmark/gnn_qa_expanded.jsonl \
  --output results/benchmark_audit_expanded.csv

python scripts/promote_verified_benchmark.py \
  --benchmark benchmark/gnn_qa_expanded.jsonl \
  --audit results/benchmark_audit_expanded.csv \
  --output benchmark/gnn_qa_verified.jsonl
```

Open the audit CSV and manually inspect a stratified sample.

**Presentation rule:** only use QA with `supported=true` and acceptable score, or manually correct unsupported items.

### Step E — smoke-test five questions before expensive runs

```bash
python scripts/run_experiments.py \
  --methods vector graph hybrid rerank full \
  --limit 5 \
  --llm-judge \
  --verify-full \
  --output results/smoke.csv
```

If a method crashes, fix it now. Do not launch the full experiment first.

### Step F — full ablation

```bash
python scripts/run_experiments.py \
  --methods vector graph hybrid rerank full \
  --llm-judge \
  --verify-full \
  --output results/ablation.csv

python scripts/summarize_results.py
python scripts/plot_results.py
```

Expected output:

```text
results/ablation.csv
results/summary.csv
results/figures/*.png
```

## 6. Methods and what each one proves

| Method | Purpose |
|---|---|
| `vector` | semantic retrieval baseline |
| `graph` | explicit relation/path baseline |
| `hybrid` | Stage-1 direct concatenation baseline |
| `rerank` | isolates value of unified evidence reranking |
| `full` | same reranking pipeline + claim/citation verification at evaluation time |

Important: `full` citation verification does not magically improve retrieval unless you add a retry loop. Present it as **trustworthiness evaluation / post-generation guard**, not as answer-quality improvement unless you actually implement regeneration.

## 7. Metrics

### Retrieval metrics

- `retrieval_recall`: fraction of gold papers present in selected evidence.
- `evidence_precision`: fraction of selected evidence whose paper belongs to gold papers.
- retrieval latency.

These are coarse paper-level metrics. They are acceptable for tomorrow, but explicitly say that passage-level gold evidence is a future improvement.

### Answer metrics

- token F1: cheap sanity metric; not sufficient alone.
- optional LLM correctness judge under the same judge prompt.
- citation validity: citations point to a paper actually present in supplied evidence.
- claim support rate: optional LLM claim-level verifier.

Do not report a single metric as “accuracy” if it is actually an LLM judge score.

## 8. Required ablation table for the PPT

Fill with real numbers only:

| Method | Retrieval Recall | Evidence Precision | LLM Correctness | Citation Validity | Retrieval Latency |
|---|---:|---:|---:|---:|---:|
| VectorRAG |  |  |  |  |  |
| GraphRAG |  |  |  |  |  |
| Vanilla HybridRAG |  |  |  |  |  |
| + Evidence Reranking |  |  |  |  |  |
| Full |  |  |  |  |  |

Then add per-question-type analysis for:

```text
Fact / Relation / Comparison / Multi-hop
```

A strong result is not “Full wins every cell”. A credible result can be:

- Vector strongest on summary/fact semantic recall.
- Graph strongest on explicit relation/multi-hop precision.
- Vanilla Hybrid improves recall but introduces evidence noise.
- Reranking restores precision while retaining hybrid recall.

That is a much better scientific story than claiming Hybrid is universally best.

## 9. Mandatory case studies

Use at least one known Stage-1 failure:

> `which model uses GCN as an encoder?`

Stage 1 Vector retrieval previously preferred GCN-heavy SGC / Cluster-GCN passages. Check whether reranking moves the VGAE evidence upward.

PPT format:

```text
Query
Stage-1 top evidence
Why it is misleading
Stage-2 reranked evidence
Why the new ranking is better
```

Also include two failures. Examples to look for:

- graph extraction relation is wrong -> reranker cannot repair incorrect KG evidence;
- section detector fails on PDF extraction artifacts;
- gold paper is absent from top candidate pool -> reranker cannot recover it.

## 10. Data-size strategy

### For tomorrow's defensible experiment

Use the existing **12-paper core corpus + audited 48–80 QA**. This is small, but controlled and reproducible.

Do **not** inflate the corpus tonight and then report unverified results.

### For final project expansion

`benchmark/gnn_paper_titles.txt` contains ~30 additional/classic GNN-related paper titles. Optional:

```bash
python scripts/fetch_gnn_corpus.py
```

This searches arXiv, downloads PDFs and produces:

```text
data/processed/papers_stage2_expanded.jsonl
```

Only switch evaluation to the expanded corpus after benchmark provenance is regenerated/reviewed.

**High standard means verified data, not merely more data.**

## 11. What NOT to say tomorrow

Do not say:

- “2-hop is our innovation.”
- “We used FAISS/BGE-M3/NetworkX, so the method is advanced.”
- “Hybrid is better because it combines two methods.”
- “The benchmark has 48 questions, therefore data is sufficient.”
- “LLM judge score is objective accuracy.”
- “Citation verification guarantees truth.”

Say instead:

> Stage 1 revealed that direct Hybrid concatenation improves evidence coverage but can introduce irrelevant context. Stage 2 therefore treats Vector chunks, graph triples and graph paths as a unified evidence pool and performs evidence-level reranking. We evaluate the contribution through controlled ablations and separately audit citation support.

## 12. Tomorrow's presentation structure

1. **Stage-1 failure analysis** — 1 slide.
2. **Research question** — 1 slide.
3. **Method architecture** — 1 slide.
4. **Evidence reranker** — 1 slide with formula/features.
5. **Benchmark & protocol** — 1 slide.
6. **Main results** — 1 slide.
7. **Per-type / ablation** — 1 slide.
8. **Case study** — 1 slide.
9. **Failure analysis** — 1 slide.
10. **Next step** — 1 slide.

Compress the old Data / Vector / Graph / Backend / Frontend implementation into backup slides.

## 13. Team allocation under a single-owner architecture

The leader owns the critical path:

```text
architecture -> reranker -> integration -> experiment -> conclusion
```

Members receive independent deliverables:

- Member A: inspect/fix section chunking outputs; deliver a 20-row audit table.
- Member B: manually review the 48 QA gold answers/provenance; deliver corrected JSONL.
- Member C: run citation/claim verifier and manually spot-check 20 answers; deliver audit CSV.
- Member D: run result summarization and prepare figures/tables from `results/ablation.csv`.

No member task is allowed to block the core pipeline.

## 14. Stop conditions tonight

Stop adding features when any of these is true:

- full 48-QA experiment has not run yet;
- benchmark has not been audited;
- reranker has no before/after case;
- main ablation table is empty;
- you cannot explain every term in the reranking score.

At that point, another feature is negative value. Run experiments and prepare the argument.
