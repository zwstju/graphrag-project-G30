# System Architecture

## 1. Project Goal

本项目面向图神经网络（GNN）科研论文，构建一个能够比较
VectorRAG、GraphRAG 和 HybridRAG 的科研论文智能问答系统。

第一阶段目标是完成完整闭环：

```text
PDF
↓
Data Processing
↓
papers.jsonl
↓
┌──────────────────────┐
↓                      ↓
Vector Retrieval       Graph Retrieval
↓                      ↓
Vector Context         Graph Context
└──────────┬───────────┘
           ↓
      Hybrid Pipeline
           ↓
      Prompt Builder
           ↓
           LLM
           ↓
       Backend API
           ↓
        Frontend
```

---

## 2. Module Ownership

| Module | Responsibility |
|---|---|
| Data Processing | PDF → cleaned `papers.jsonl` |
| Vector Retrieval | Chunking, embedding, vector index, `retrieve_vector()` |
| Graph Retrieval | KG construction, graph retrieval, `retrieve_graph()` |
| HybridRAG / Backend | Query processing, fusion, prompt, LLM, `/ask` |
| Frontend | UI, mode selection, evidence display |

---

## 3. Shared Data

所有下游模块都以：

```text
data/processed/papers.jsonl
```

作为公共数据入口，并共享相同 `paper_id`。

---

## 4. Online Query Flow

```text
User Query
   ↓
Query Processor
   ↓
Intent + Entity Recognition
   ↓
┌─────────────────┬─────────────────┐
↓                                   ↓
Vector Retriever                    Graph Retriever
↓                                   ↓
Top-k Chunks                         Triples / Paths
└─────────────────┬─────────────────┘
                  ↓
             Hybrid Fusion
                  ↓
             Prompt Builder
                  ↓
             Unified LLM
                  ↓
               Answer
```

当前 HybridRAG baseline 使用最简单、最容易复现的融合：

```text
Vector Context
+
Graph Context
↓
Concatenate
↓
LLM
```

---

## 5. Comparison Modes

Backend 对外暴露三种模式：

```text
vector
graph
hybrid
```

最终实验在同一数据、同一问题集和尽量相同生成模型下比较三种方法。

---

## 6. Stage-2 Upgrade

在基础 HybridRAG 跑通后，重点升级 Graph Retriever：

```text
Query
↓
Entity Recognition
↓
Seed Nodes
↓
Query Intent
↓
Dynamic 1~2 Hop Expansion
↓
Candidate Paths
↓
Path Relevance Scoring
↓
Top-k Paths
↓
Graph Context
```

目标是研究：

> 不同图检索策略对关系问题和多跳问题的影响。

最终可比较：

```text
VectorRAG
Basic GraphRAG
HybridRAG baseline
HybridRAG + Query-Adaptive Graph Retrieval
```

---

## 7. Explainability

Frontend 除最终答案外，应尽量显示：

```text
Vector Evidence
Graph Triples / Reasoning Paths
Paper References
```

例如：

```text
GCN ←USES— VGAE —SOLVES→ Link Prediction
```

从而让用户看到答案所依据的图结构证据。
