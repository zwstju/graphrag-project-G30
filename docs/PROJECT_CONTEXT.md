# Project Context

## 1. Project

Title:

**Evidence-Centric HybridRAG for Graph Neural Network Scientific Literature QA**

The project is for a university Knowledge Engineering course.

The system answers questions over Graph Neural Network research papers.

---

## 2. Current Corpus

The controlled core corpus contains 12 papers:

1. `gcn_2017`
   Semi-Supervised Classification with Graph Convolutional Networks

2. `vgae_2016`
   Variational Graph Auto-Encoders

3. `graphsage_2017`
   Inductive Representation Learning on Large Graphs

4. `gat_2018`
   Graph Attention Networks

5. `gin_2019`
   How Powerful are Graph Neural Networks?

6. `deepwalk_2014`
   DeepWalk: Online Learning of Social Representations

7. `node2vec_2016`
   node2vec: Scalable Feature Learning for Networks

8. `mpnn_2017`
   Neural Message Passing for Quantum Chemistry

9. `diffpool_2018`
   Hierarchical Graph Representation Learning with Differentiable Pooling

10. `sgc_2019`
    Simplifying Graph Convolutional Networks

11. `appnp_2019`
    Predict then Propagate: Graph Neural Networks meet Personalized PageRank

12. `cluster_gcn_2019`
    Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks

Canonical corpus:

`data/processed/papers.jsonl`

---

## 3. Stage 1 Baseline

### Vector retrieval

Stage 1 used:

- BGE-M3
- fixed-size character chunks
- overlap
- normalized embeddings
- FAISS `IndexFlatIP`

There were approximately 596 fixed-size Vector chunks.

Known problem:

BGE-M3 + semantic similarity can retrieve passages that are lexically/semantically related but do not logically answer relation questions.

Example failure pattern:

Question:

`Which model uses GCN as an encoder?`

Pure Vector retrieval may rank general GCN-related passages from SGC or Cluster-GCN,
while the VGAE paper contains the directly relevant evidence.

---

### Graph retrieval

Knowledge graph is extracted with an LLM.

Pipeline:

paper text
→ chunks
→ LLM refinement
→ entity/relation extraction
→ triples
→ entity resolution
→ NetworkX MultiDiGraph

Current graph is approximately:

- 1204 nodes
- 1761 edges

Canonical triples:

`data/graph/triples.jsonl`

Stage 1 graph retrieval mainly uses:

- lexical entity linking;
- fixed 1-hop expansion;
- rule-based triple scoring;
- shortest paths between query seed entities.

Entity resolution currently uses:

- text normalization;
- normalized comparison keys;
- manually defined aliases;
- duplicate-triple removal.

It is a rule-based baseline, not a learned entity-resolution system.

---

### Vanilla HybridRAG

Stage 1 HybridRAG combines:

Vector evidence
+
Graph evidence
→ context concatenation
→ LLM

There is no learned fusion mechanism.

This is retained as an experimental baseline.

---

## 4. Stage 2 Research Motivation

The central problem is no longer:

"How do we combine VectorRAG and GraphRAG?"

Instead:

"How do we select useful scientific evidence from heterogeneous retrievers?"

Vanilla HybridRAG can improve coverage while lowering context precision because irrelevant evidence is also added.

Stage 2 therefore uses an evidence-centric design.

---

## 5. Stage 2 Architecture

Conceptual pipeline:

Query
↓
Vector Candidate Retrieval
+
Graph Triple / Path Retrieval
↓
Unified Evidence Pool
↓
Scientific Evidence Reranking
↓
Evidence Selection
↓
Grounded LLM Generation
↓
Citation / Claim Verification
↓
Evaluation

Relevant code lives under:

`stage2/`

---

## 6. Structure-Aware Chunking

The original fixed-character chunker was replaced for Stage 2 retrieval experiments.

The current structure-aware chunker:

1. recognizes broad scientific-paper sections;
2. maps noisy/fine-grained PDF headings into canonical section classes;
3. chunks inside sections;
4. prefers paragraph and sentence boundaries;
5. only uses length-based fallback when necessary;
6. avoids cutting words in half.

Canonical sections currently include:

- Abstract
- Introduction
- Related Work
- Background
- Method
- Experiments
- Results
- Conclusion
- Appendix
- References
- Unknown

Latest actual output:

- total chunks: 488
- Experiments: 202
- Method: 140
- Introduction: 44
- References: 42
- Related Work: 14
- Abstract: 12
- Conclusion: 11
- Results: 9
- Appendix: 8
- Background: 4
- Unknown: 2

Length statistics:

- minimum: 83 characters
- median: 1515.5 characters
- mean: approximately 1259.9 characters
- maximum: 1800 characters

Current file:

`data/stage2/structure_chunks.jsonl`

For the main Vector retrieval index, References should be excluded.

Expected retrieval chunk count after excluding 42 References chunks:

488 - 42 = 446

Do not assume 446 until the filtered file is actually generated and checked.

---

## 7. Benchmark

Main controlled benchmark:

`benchmark/gnn_qa_core80.jsonl`

Current candidate benchmark statistics:

- total: 80
- Fact: 20
- Relation: 20
- Comparison: 20
- Multi-hop: 20

Difficulty:

- easy: 12
- medium: 40
- hard: 28

Gold-paper coverage:

- 12 / 12 core papers

Paper-ID validation has already succeeded:

All benchmark gold-paper IDs exist in the core corpus.

Important:

These questions are currently candidate/model-authored benchmark items.

They must undergo evidence audit and representative manual inspection before being described as verified gold data.

---

## 8. Expanded Data

The repository also contains expansion resources:

- `benchmark/gnn_papers_expanded.jsonl`
- `benchmark/gnn_paper_titles.txt`
- `benchmark/gnn_qa_expanded.jsonl`

The target expansion is roughly 50 scientific papers.

Do NOT claim that all 50 paper full texts have already been ingested unless this is actually completed and verified.

For the immediate controlled experiment, prefer the 12-paper Core80 setting so Vector / Graph / Hybrid methods operate on the same corpus.

---

## 9. Evidence Reranking

The initial Stage 2 reranker combines interpretable signals such as:

- semantic relevance;
- entity overlap;
- relation match;
- scientific section prior;
- evidence source/type;
- diversity.

Conceptually:

S(e,q) =
alpha * semantic
+ beta * entity
+ gamma * relation
+ delta * section
+ other evidence features

This rule-based reranker is intended as an interpretable baseline.

A neural Cross-Encoder reranker may be added later if the controlled pipeline is already stable.

Do not add a neural reranker before baseline experiments run correctly.

---

## 10. Citation Verification

The Stage 2 pipeline includes answer/evidence verification.

Goal:

answer
→ claims
→ retrieved evidence
→ determine whether each important claim is supported

Possible output:

- total claims;
- supported claims;
- support rate;
- supporting evidence IDs.

LLM-based verification is an evaluator and should not be treated as infallible ground truth.

Manual spot checking is required.

---

## 11. Experimental Design

Required methods:

1. VectorRAG
2. GraphRAG
3. Vanilla HybridRAG
4. Hybrid + Evidence Reranking
5. Full Stage 2 System

Desired result table:

| Method | Retrieval Recall | Evidence Precision | Answer Correctness | Citation Support | Latency |
|---|---:|---:|---:|---:|---:|
| VectorRAG | measured | measured | measured | measured | measured |
| GraphRAG | measured | measured | measured | measured | measured |
| Vanilla Hybrid | measured | measured | measured | measured | measured |
| + Rerank | measured | measured | measured | measured | measured |
| Full | measured | measured | measured | measured | measured |

Never fill this table with invented numbers.

Negative or mixed results are acceptable and should be analyzed.

---

## 12. Hardware and Environment

Current environment:

- Arch Linux
- Python virtualenv `.venv`
- NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM approximately 7.65 GB
- PyTorch `2.14.0+cu130`
- `torch.cuda.is_available() == True`

Use CUDA for BGE-M3 embeddings and neural models.

Start BGE-M3 embedding batch size around 16.

---

## 13. Current Exact State

Completed:

- Stage 1 system exists.
- Stage 1 Vector / Graph / Hybrid pipeline has previously worked.
- Core corpus contains 12 papers.
- Graph triples already exist.
- Core80 benchmark paper IDs have been validated.
- Structure-aware chunking has been rebuilt.
- Latest structure-aware corpus contains 488 chunks.
- Section normalization has been improved.
- Accidental shell wrapper lines were removed from `AGENTS.md` and this file.
- `data/stage2/retrieval_chunks.jsonl` was generated in source order.
- Retrieval-corpus validation reports 488 source chunks, 42 excluded canonical
  `References` chunks, 446 output chunks, and 0 remaining `References` chunks.
- `configs/default.json` now keeps structure chunks and retrieval chunks under
  separate `chunks_path` and `retrieval_chunks_path` fields.
- Stage 2 vector indexing now supports explicit `auto|cuda|cpu` device selection,
  an explicit default embedding batch size of 16, normalized embeddings, and
  FAISS `IndexFlatIP` persistence.
- The vector build writes an integrity manifest containing corpus/index/metadata
  hashes, model, chunk count, embedding dimension, normalization state, index
  type, paths, device, batch size, and build timestamp.
- Runtime index loading validates index/metadata count, manifest hashes, model,
  dimension, normalization, index type, and current retrieval-corpus hash. It
  loads the persisted index and does not re-embed the corpus.
- A dedicated five-query qualitative vector smoke-test command is available.
- Non-GPU tests for device resolution and index-integrity failure modes pass.
- Stage 2 experiment outputs now identify structured/improved retrieval
  conditions explicitly; paper-level retrieval metrics are labeled as proxies.
- The persistent Stage 2 BGE-M3 index was successfully built in the real host
  environment using CUDA on the NVIDIA GeForce RTX 4060 Laptop GPU.
- The verified index contains 446 normalized 1024-dimensional embeddings in a
  FAISS `IndexFlatIP`; `index.ntotal == metadata_count == 446`, and the indexed
  retrieval corpus contains 0 canonical `References` chunks.
- The five-query qualitative Vector retrieval smoke test completed successfully.
- For the VGAE/GCN relation query, VGAE evidence ranked first and second.
- For `Which model uses GCN as an encoder?`, VGAE evidence ranked first.
- For the GraphSAGE dataset query, GraphSAGE `Experiments` evidence ranked first.
- The GCN/GAT neighbor-weight comparison exposed a useful Vector failure: APPNP
  ranked first while GAT ranked only second.
- For the GCN-encoder + adjacency-reconstruction + link-prediction query, VGAE
  ranked first, second, and fifth, but irrelevant GCN-related evidence remained.
- The benchmark audit now uses gold-aware evidence mining for benchmark
  validation over `data/stage2/retrieval_chunks.jsonl`; it is explicitly not a
  retrieval-evaluation procedure.
- A fixed balanced Core80 audit smoke subset contains 8 items: 2 Fact, 2
  Relation, 2 Comparison, and 2 Multi-hop, including 1-, 2-, and 3-paper items.
- The audit smoke selected 3 chunks independently from every declared gold paper.
  All 8 items met the per-paper quota, and no selected evidence came from
  `References`.
- The canonical smoke audit artifact is `results/core80_audit_smoke8.jsonl`, with
  a compact CSV and a run manifest containing input/model/index hashes and audit
  configuration.
- Smoke audit outcome: 7 items are `llm_evidence_supported`; `CR17` is
  `llm_evidence_not_supported` because the selected paper evidence does not
  directly establish that GCN and GAT are MPNN special cases.
- All 8 smoke items remain `manual_status=unreviewed`. Automatic audit changed 0
  candidate gold answers; suggestions are stored separately.
- The full Core80 LLM evidence audit has completed over all 80 candidate items
  using CUDA, gold-aware benchmark validation, 3 selected chunks per declared
  gold paper, a 6000-character cap per paper, and at most 3 LLM attempts.
- The canonical detailed artifact is `results/core80_audit.jsonl`; companion
  artifacts are `results/core80_audit.csv`,
  `results/core80_audit.manifest.json`,
  `results/core80_audit_summary.json`, and
  `results/core80_manual_review_queue.jsonl`.
- Full audit outcome: 71 items were successfully audited, of which 69 are
  `llm_evidence_supported` and 2 are `llm_evidence_not_supported`; 9 items ended
  as `llm_audit_error` after all retries. The candidate-answer evidence support
  rate among successfully audited items is 69/71 (97.18%). Mean and median
  support scores are 0.9675 and 1.0, respectively.
- The unsupported QIDs are `CF17` and `CC14`. Suggested answers are retained only
  as manual-review proposals; none was applied to the candidate benchmark.
- The run made 111 total LLM API attempts, including 31 retries. It recorded 38
  empty responses, 2 JSON parse failures, 0 transport failures, and 9 items that
  ultimately failed after all 3 attempts. Infrastructure failures remain
  distinct from semantic `llm_evidence_not_supported` judgments.
- Core80 contains 50 single-paper, 27 two-paper, and 3 three-paper questions.
  All 30 multi-paper questions received evidence from every declared gold
  paper, with 0 omission violations and 0 per-paper quota violations.
- The full audit selected 339 evidence chunks. Referential-integrity checks found
  0 `References` chunks, 0 missing chunk IDs, 0 paper-ID mismatches, 0 candidate
  gold-answer modifications, and only `unreviewed` manual statuses.
- Exact LLM audit runtime was 547.577442234 seconds, from
  `2026-09-14T13:05:27.090734+00:00` through
  `2026-09-14T13:14:34.668128+00:00`.
- The automatic evaluation benchmark is now frozen as the 69-item
  `Core69 Audit-Filtered Evaluation Set` at
  `benchmark/gnn_qa_core69_audit_filtered.jsonl`. Inclusion is based only on
  `llm_evidence_supported`, not human verification; all original Core80
  questions and candidate gold answers are preserved.
- Core69 contains 17 Fact, 17 Relation, 16 Comparison, and 19 Multi-hop items;
  difficulty counts are 11 easy, 33 medium, and 25 hard. It covers all 12 core
  gold papers and has SHA-256
  `804da496df045bb1c7e792701eb56586c1914b7c9c6bbefab3e4606626e6b26c`.
- Targeted Graph retrieval fixes now use token/boundary-aware entity matching,
  enforce strict `top_k`, deterministically score parallel relations, preserve
  all path paper IDs, and exclude `WRITTEN_BY`/`CITES` from normal QA retrieval.
- The rule-based reranker now infers qtype from query text only, tokenizes mixed
  Chinese/English queries into useful features, scores actual relation/content,
  retains normalized retrieval score as an interpretable component, and stores
  every score part on each Evidence item.
- Query-text-only qtype inference matches 80/80 Core80 annotations in the frozen
  evaluation (`results/final/qtype_classifier_eval.json`). This is an in-domain
  benchmark result for transparent rules, not evidence of general classifier
  accuracy outside Core80.
- A separate fixed-character control corpus and index now exist at
  `data/stage2/fixed_chunks_1200_200.jsonl` and
  `data/stage2/fixed_vector_1200_200.index`. The corpus contains 521 chunks with
  1200-character windows and 200-character overlap after the shared References
  exclusion policy. Its CUDA-built BGE-M3 index contains 521 normalized
  1024-dimensional vectors in FAISS `IndexFlatIP` and does not overwrite the
  446-chunk structured index.
- The four-condition Core69 retrieval ablation completed over all 69 questions,
  with exactly 6 final Evidence items per condition. For every question,
  Structured Vanilla Hybrid and Structured Hybrid + Rerank used identical saved
  12-Vector + 12-Graph candidate IDs.
- Overall paper-level proxy recall / precision were: Fixed Vector
  0.8237 / 0.7536; Structured Vector 0.8671 / 0.7585; Structured Vanilla Hybrid
  0.8575 / 0.7440; Structured Hybrid + Rerank 0.8599 / 0.7560. These are
  paper-provenance proxies, not passage-level relevance judgments.
- Mean retrieval latency was 0.0151 s for Fixed Vector, 0.0120 s for Structured
  Vector, 0.0257 s for Vanilla Hybrid, and 1.4836 s for Hybrid + Rerank. The
  rule-based reranker's additional BGE-M3 evidence scoring is the dominant cost.

Environment distinction:

- The real local Arch/VS Code terminal has verified CUDA and the NVIDIA GeForce
  RTX 4060 Laptop GPU.
- The default Codex sandbox does not expose CUDA, while the approved host-level
  audit execution did. This is an execution-context distinction, not evidence
  that host CUDA is broken.

Known experiment blockers (documented, not fixed in the audit milestone):

- Graph substring seed matching can confuse `SGC` with text such as `uses GCN`.
- Graph paths can make returned evidence exceed `top_k`.
- MultiDiGraph path edge selection is not deterministic by explicit policy.
- Multi-paper graph paths expose incomplete top-level paper provenance.
- `WRITTEN_BY` and `CITES` relations add substantial graph noise.
- Reranker-inferred question types disagree with many Core80 labels.
- Chinese entity-overlap tokenization is weak.
- The reranker does not use the original retrieval score.
- The current conditions do not isolate structure-aware chunking from all other
  changes.
- Verification is post-hoc and does not feed back into generation.

Not yet considered verified/completed:

- manual verification of Core80 audit items;
- full 5-method controlled experiment;
- final ablation table;
- final experiment figures;
- neural Cross-Encoder comparison.

---

## 14. Immediate Next Task

The immediate engineering task is:

1. Complete the deterministic Core32 generation evaluation: 8 items per qtype,
   three methods, identical LLM settings, 6 saved Evidence items per answer, and
   infrastructure failures retained separately from evaluation scores.
2. Run post-hoc claim verification over the exact saved Hybrid + Rerank answers
   without regeneration.
3. Aggregate measured retrieval, generation, citation, and latency results into
   final tables and figures.
4. Finalize the four required case studies with at least two measured
   improvements and two honestly reported remaining failures.

Do not introduce unrelated functionality before this sequence succeeds.

---

## 15. Working Philosophy

The project should be treated as a research prototype.

Always ask:

- What problem does this change solve?
- What baseline does it improve over?
- How will we measure the improvement?
- What failure mode remains?
- Is the claim supported by experiment?

Do not confuse:
"it runs"
with
"it works".

Do not confuse:
"more modules"
with
"more novelty".
