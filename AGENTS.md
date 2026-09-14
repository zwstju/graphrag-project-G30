# AGENTS.md

## Role

You are working on a university knowledge-engineering project:

**Evidence-Centric HybridRAG for Graph Neural Network Scientific Literature QA**

Act as a strict research engineer, not merely a code generator.

Before making substantial changes:

1. Read `docs/PROJECT_CONTEXT.md`.
2. Read `README_STAGE2.md`.
3. Read `PROJECT_REVIEW_CHECKLIST.md`.
4. Inspect the relevant existing source files.
5. Preserve working baselines unless there is a concrete reason to change them.

Do not redesign the whole project unless the current design is demonstrably flawed.

---

## Repository Architecture

### Stage 1 baseline

`src/` contains the original HybridRAG baseline.

Important components:

- `src/vector/`
  - BGE-M3 embeddings
  - FAISS vector retrieval
- `src/graph/`
  - LLM knowledge extraction
  - entity resolution
  - NetworkX graph retrieval
  - triple/path retrieval
- `src/hybrid/`
  - vanilla Vector + Graph fusion
- `src/backend/`
  - FastAPI backend
- `frontend/`
  - Web interface

Do NOT destroy or silently rewrite the Stage 1 baseline.
It is required for controlled ablation experiments.

### Stage 2

`stage2/` contains the new Evidence-Centric HybridRAG method.

Main ideas:

1. Structure-aware scientific-paper chunking
2. Vector + Graph heterogeneous evidence retrieval
3. Unified Evidence representation
4. Evidence reranking
5. Evidence selection
6. Grounded answer generation
7. Citation / claim verification
8. Controlled ablation evaluation

---

## Canonical Data

Use these as canonical sources:

- Scientific papers:
  `data/processed/papers.jsonl`

- Knowledge-graph triples:
  `data/graph/triples.jsonl`

- Stage 2 structure-aware chunks:
  `data/stage2/structure_chunks.jsonl`

- Main controlled benchmark:
  `benchmark/gnn_qa_core80.jsonl`

Do not silently replace these paths.

---

## Experimental Integrity

Never fabricate experimental results.

Do not claim an improvement unless it is supported by actual experiment output.

Candidate benchmark questions are NOT automatically verified gold data.

Before reporting benchmark results:

- validate paper IDs;
- evidence-audit benchmark items;
- manually inspect representative samples;
- report limitations honestly.

Keep experimental conditions controlled:
same corpus, same question set, same generation model, same evaluation protocol unless the experiment explicitly studies that variable.

---

## Hardware

Local machine:

- NVIDIA GeForce RTX 4060 Laptop GPU
- VRAM: approximately 7.65 GB
- PyTorch: 2.14.0+cu130
- CUDA available: True

Use CUDA for:

- BGE-M3 embedding generation
- neural rerankers
- model training / inference where appropriate

Do not waste GPU resources on trivial JSON, NetworkX, or small FAISS operations.

For BGE-M3 start with batch size 16.
Increase only after confirming VRAM stability.

---

## Python Execution

The repository root should be treated as the Python project root.

Prefer module-style execution when imports depend on project packages:

    python -m scripts.<script_name_without_py>

Do not introduce fragile `sys.path` hacks unless necessary.

---

## Current Research Question

Stage 1 exposed a key limitation:

Vanilla HybridRAG increases evidence coverage by combining Vector and Graph retrieval,
but direct concatenation can introduce semantically similar yet logically irrelevant evidence.

Stage 2 asks:

**Can structure-aware retrieval, heterogeneous evidence reranking, and citation verification improve scientific QA evidence quality and answer reliability?**

The project contribution is NOT merely "VectorRAG + GraphRAG".

---

## Required Baselines

Preserve and compare:

1. VectorRAG
2. GraphRAG
3. Vanilla HybridRAG
4. HybridRAG + evidence reranking
5. Full Stage 2 system

Metrics should include, where available:

- Retrieval Recall
- Evidence Precision
- Answer Correctness
- Citation Support / Validity
- Retrieval or end-to-end latency

Perform ablation studies rather than only showing demos.

---

## Immediate Current Priority

Do NOT add unrelated features.

Current sequence:

1. Finish and validate structure-aware retrieval corpus.
2. Exclude References chunks from the main retrieval index.
3. Ensure BGE-M3 embedding generation actually uses CUDA.
4. Persist embeddings / FAISS index.
5. Run a small smoke test.
6. Run controlled Core80 experiments.
7. Produce ablation tables and figures.
8. Inspect successful and failed cases.
9. Only then consider neural reranking or additional improvements.

When a command or experiment fails, diagnose the existing implementation before replacing it wholesale.

---

## Engineering Standard

For every meaningful code change:

- explain why the change is needed;
- identify which hypothesis it supports;
- preserve reproducibility;
- provide a command to validate it;
- inspect outputs rather than treating successful execution as proof of correctness.

Do not confuse engineering complexity with research novelty.

The goal is a defensible experimental project, not the largest codebase.
