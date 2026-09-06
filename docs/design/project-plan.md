# Project Plan

## Stage 1: Minimal Working System

目标：先完成完整闭环，不在第一阶段过度优化算法。

### Data
- 处理 10~20 篇 GNN 核心论文
- 输出统一 `papers.jsonl`
- 保证所有论文有唯一 `paper_id`

### VectorRAG
- Chunking
- Embedding
- Vector Index
- `retrieve_vector(query, top_k)`

### GraphRAG
- Entity / Relation Extraction
- Knowledge Triples
- Knowledge Graph
- `retrieve_graph(query, top_k)`

### Hybrid / Backend
- Query Processor
- Vector / Graph / Hybrid 三模式
- Prompt Builder
- Unified LLM Client
- `POST /ask`

### Frontend
- Query Input
- Mode Selector
- Final Answer
- Vector Evidence
- Graph Evidence
- References

---

## Stage 2: Paper Reproduction and Improvement

### Baselines
1. VectorRAG
2. Basic GraphRAG
3. HybridRAG

### Main Improvement
Query-Adaptive Multi-hop Graph Retrieval

```text
Query
↓
Entities / Intent
↓
Dynamic Neighbor Expansion
↓
Candidate Paths
↓
Ranking
↓
Top-k Graph Evidence
```

---

## Evaluation

问题分为：

```text
Fact
Relation
Multi-hop
Summary
```

重点观察：

- VectorRAG 更适合哪些问题？
- GraphRAG 更适合哪些问题？
- HybridRAG 是否能利用互补信息？
- Query-aware graph retrieval 是否改善关系 / 多跳问题？

建议记录：

```text
question
ground_truth
mode
retrieved_context
generated_answer
manual_score / metric
```

最终展示三种或四种方法的对比结果，而不是只展示一个 Demo。
