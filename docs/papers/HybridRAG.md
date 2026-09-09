# HybridRAG Reading Notes

论文：

**HybridRAG: Integrating Knowledge Graphs and Vector Retrieval Augmented Generation for Efficient Information Extraction**

---

## 1. Core Problem

论文关注从复杂、非结构化金融文档中进行信息抽取和问答。

作者认为单独使用 VectorRAG 时，文本切块和向量相似度检索可能无法充分保留文档中的结构、关系和领域上下文。

GraphRAG 可以利用知识图谱中的实体和关系，但在抽象问题或问题中没有显式实体时可能表现较弱。

因此论文提出：

```text
VectorRAG
+
GraphRAG
=
HybridRAG
```

核心思想不是重新训练一个新的大模型，而是联合两种不同来源的检索上下文。

---

## 2. VectorRAG

论文中的 VectorRAG 流程：

```text
Documents
↓
Chunking
↓
Embedding
↓
Vector Database
↓
Query Similarity Search
↓
Top Chunks
↓
Context
↓
LLM
```

实现部分使用：

- Pinecone Vector Database
- OpenAI `text-embedding-ada-002`
- GPT-3.5-Turbo
- LangChain

论文实验配置中：

```text
Chunk Size: 1024
Chunk Overlap: 0
Similarity Candidate Chunks: 20
Retrieved Contexts: 4
```

VectorRAG 的主要优势是：

> 通过语义相似度获得较广泛的文本上下文。

---

## 3. Knowledge Graph Construction

论文首先将 PDF 文档进行预处理和分块。

知识图谱构建采用两级 LLM Chain：

```text
Document Chunk
↓
LLM 1
内容提炼 / Abstract Representation
↓
LLM 2
Entity + Relation Extraction
↓
Knowledge Triples
```

作者的目标是先压缩 chunk 中的关键信息，再进行实体和关系抽取。

论文实现部分还提到：

```text
PyPDFLoader
RecursiveCharacterTextSplitter
```

用于 PDF 读取和分块。

知识三元组包含：

```text
Head Entity
Head Type
Relation
Object Entity
Object Type
Metadata
```

---

## 4. GraphRAG

GraphRAG 与 VectorRAG 的主要区别位于 Retrieval。

```text
Query
↓
Knowledge Graph Retrieval
↓
Relevant Nodes + Edges
↓
Subgraph / Graph Context
↓
LLM
```

论文实现使用：

```text
Knowledge Triples
↓
NetworkX / NetworkxEntityGraph
↓
GraphQAChain
↓
Depth-First Search
↓
Relevant Graph Context
```

一个重要实现细节：

```text
DFS Depth = 1
```

因此论文中的 Graph Retriever 并不是非常复杂的多跳图算法。

这使它非常适合作为我们项目中的 Basic GraphRAG baseline。

---

## 5. HybridRAG

论文 HybridRAG 的核心实现非常直接。

```text
VectorRAG Context
+
GraphRAG Context
↓
Concatenate
↓
Unified Context
↓
LLM
↓
Answer
```

原论文的顺序是：

```text
Vector Context
↓
Graph Context
```

即 VectorRAG 检索到的上下文放在前面，GraphRAG 上下文随后追加。

因此第一版复现可以实现：

```python
vector_results = retrieve_vector(query)
graph_results = retrieve_graph(query)

prompt = build_prompt(
    query,
    vector_results,
    graph_results
)

answer = generate_answer(prompt)
```

不需要设计复杂的融合网络即可完成论文核心思路的复现。

---

## 6. Evaluation

论文分别评价 Retrieval 和 Generation。

主要指标包括：

### Faithfulness
生成答案中的陈述是否能够由检索上下文支持。

### Answer Relevance
生成答案是否真正回答了用户的问题。

### Context Precision
检索得到的上下文中，有多少是真正相关的。

### Context Recall
Ground Truth 中的信息有多少能够在检索上下文中找到。

论文报告：

| Method | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---:|---:|---:|---:|
| VectorRAG | 0.94 | 0.91 | 0.84 | 1.00 |
| GraphRAG | 0.96 | 0.89 | 0.96 | 0.85 |
| HybridRAG | 0.96 | 0.96 | 0.79 | 1.00 |

值得注意的是，HybridRAG 的 Context Precision 并不是最高。

原因是两种 context 拼接后会引入更多信息，其中一些并不直接对应 Ground Truth。

这意味着我们的项目不应该简单声称：

> HybridRAG 在所有指标上都一定更好。

更合理的研究问题是：

> Vector 和 Graph 两种检索在不同问题类型上有什么互补性？

---

## 7. Paper Observation

论文观察到：

- GraphRAG 更适合一些显式实体和抽取式问题。
- GraphRAG 在问题没有明确实体时可能失败。
- VectorRAG 在更抽象的问题中可能更有优势。
- HybridRAG 通过同时保留两种 context，在整体上取得更平衡的效果。

这正好支持我们的实验设计：

```text
Fact
Relation
Multi-hop
Summary
```

分别观察三种 RAG 的表现。

---

# 8. Mapping to Our Project

原论文：

```text
Financial Earnings Calls
```

我们的项目：

```text
GNN Research Papers
```

对应关系：

| HybridRAG Paper | Our Project |
|---|---|
| Financial Documents | GNN Papers |
| Company / Executive / Metric | Model / Method / Dataset / Task / Author |
| Vector Database | Vector Index |
| Financial KG | GNN Literature KG |
| VectorRAG | VectorRAG Baseline |
| GraphRAG | GraphRAG Baseline |
| HybridRAG | HybridRAG Reproduction |

---

# 9. Our Reproduction

第一阶段：

```text
papers.jsonl
↓
┌────────────────────┐
↓                    ↓
VectorRAG             GraphRAG
↓                    ↓
Vector Context        Graph Context
└─────────┬──────────┘
          ↓
     Concatenation
          ↓
          LLM
          ↓
        Answer
```

这一版本基本对应原论文的 HybridRAG 核心方法。

---

# 10. Our Improvement

原论文 GraphRAG 使用较简单的深度 1 图检索。

因此项目第二阶段可以把创新集中在 Graph Retrieval：

```text
Query
↓
Entity Recognition
↓
Query Intent
↓
Seed Nodes
↓
Dynamic 1~2 Hop Expansion
↓
Candidate Graph Paths
↓
Path Ranking
↓
Top-k Paths
```

暂定名称：

**Query-Adaptive Graph Retrieval**

最终比较：

```text
VectorRAG
vs
Basic GraphRAG
vs
HybridRAG
vs
HybridRAG + Query-Adaptive Graph Retrieval
```

重点研究：

> 更好的 Graph Retrieval 是否能够改善 GNN 论文中的关系问题和多跳问题？
