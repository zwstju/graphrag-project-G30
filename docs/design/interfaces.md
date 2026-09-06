# Module Interface Specification

本文档定义 GraphRAG 项目各模块之间的输入、输出与调用方式。

所有成员应按照本文件规定的接口进行开发。

如果需要修改接口字段、文件格式或函数签名，应先在组内同步，避免模块集成时出现不兼容问题。

---

# 1. Overall Pipeline

系统整体数据流如下：

```text
Raw PDF Papers
      ↓
Data Processing
      ↓
papers.jsonl
      ↓
┌───────────────────────┐
│                       │
↓                       ↓
Vector Pipeline       Graph Pipeline
↓                       ↓
Vector Retriever      Graph Retriever
│                       │
└───────────┬───────────┘
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

# 2. Data Processing Interface

负责人：Data Processing

## 2.1 Input

原始论文 PDF：

```text
data/raw/
├── gcn_2017.pdf
├── vgae_2016.pdf
├── gat_2018.pdf
├── graphsage_2017.pdf
├── gin_2019.pdf
└── ...
```

## 2.2 Output

统一处理后的论文数据：

```text
data/processed/papers.jsonl
```

采用 JSONL 格式：

- 一行表示一篇论文
- 每篇论文必须具有唯一 `paper_id`
- Vector 和 Graph 模块必须使用相同的 `paper_id`

### Required Schema

```json
{
  "paper_id": "vgae_2016",
  "title": "Variational Graph Auto-Encoders",
  "authors": [
    "Thomas N. Kipf",
    "Max Welling"
  ],
  "year": 2016,
  "abstract": "Paper abstract...",
  "text": "Cleaned full paper text..."
}
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `paper_id` | `string` | 论文唯一 ID |
| `title` | `string` | 论文标题 |
| `authors` | `list[string]` | 作者列表 |
| `year` | `int` | 发表年份 |
| `abstract` | `string` | 论文摘要 |
| `text` | `string` | 清洗后的论文正文 |

### Example

```json
{
  "paper_id": "gcn_2017",
  "title": "Semi-Supervised Classification with Graph Convolutional Networks",
  "authors": [
    "Thomas N. Kipf",
    "Max Welling"
  ],
  "year": 2017,
  "abstract": "...",
  "text": "..."
}
```

## 2.3 Data Processing Requirements

Data Processing 模块不能只提交处理后的数据，还必须提交能够重复执行的数据处理代码。

例如：

```bash
python src/data_processing/build_dataset.py
```

运行后应能够完成：

```text
data/raw/*.pdf
        ↓
PDF Parsing
        ↓
Text Cleaning
        ↓
Metadata Extraction
        ↓
data/processed/papers.jsonl
```

---

# 3. Vector Retriever Interface

负责人：Vector RAG

Vector 模块内部实现方式可以自由选择，例如：

- FAISS
- Chroma
- BGE Embedding
- OpenAI Embedding
- 其他 Embedding Model / Vector Database

但对外必须遵守本节规定的接口。

---

## 3.1 Build Vector Index

输入：

```text
data/processed/papers.jsonl
```

内部流程：

```text
papers.jsonl
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Database / Index
```

每个 Chunk 至少应保存以下信息：

```json
{
  "paper_id": "vgae_2016",
  "chunk_id": "vgae_2016_003",
  "text": "The encoder of VGAE is based on a graph convolutional network...",
  "section": "Method"
}
```

其中：

- `paper_id`：必须存在
- `chunk_id`：必须存在且唯一
- `text`：必须存在
- `section`：推荐保留，可以暂时为空

---

## 3.2 Vector Retrieval Function

Vector 模块必须提供：

```python
def retrieve_vector(
    query: str,
    top_k: int = 5
) -> list[dict]:
    ...
```

### Input Example

```python
query = "What is the relationship between VGAE and GCN?"
top_k = 5
```

### Output Example

```json
[
  {
    "paper_id": "vgae_2016",
    "chunk_id": "vgae_2016_003",
    "text": "The encoder of VGAE is based on a graph convolutional network...",
    "score": 0.91
  },
  {
    "paper_id": "gcn_2017",
    "chunk_id": "gcn_2017_006",
    "text": "Graph convolutional networks propagate information between neighboring nodes...",
    "score": 0.84
  }
]
```

### Required Fields

每个检索结果必须包含：

| Field | Type | Description |
|---|---|---|
| `paper_id` | `string` | 来源论文 ID |
| `chunk_id` | `string` | Chunk 唯一 ID |
| `text` | `string` | 检索得到的文本 |
| `score` | `float` | 相似度或检索得分 |

可以额外返回：

```json
{
  "title": "Variational Graph Auto-Encoders",
  "section": "Method"
}
```

---

# 4. Graph Retriever Interface

负责人：Knowledge Graph / Graph Retrieval

Graph 模块内部可以使用：

- Neo4j
- Cypher
- LLM Entity Extraction
- LLM Relation Extraction
- Dynamic Graph Retrieval
- Multi-hop Retrieval

Backend 不应该依赖 Graph 模块内部的具体实现方式。

---

## 4.1 Knowledge Graph Construction

输入：

```text
data/processed/papers.jsonl
```

内部流程：

```text
papers.jsonl
    ↓
Entity Extraction
    ↓
Relation Extraction
    ↓
Knowledge Triples
    ↓
Knowledge Graph
```

建议实体类型：

```text
Paper
Model
Method
Dataset
Task
Author
Concept
```

建议关系类型：

```text
PROPOSES
USES
EVALUATED_ON
SOLVES
EXTENDS
SIMPLIFIES
WRITTEN_BY
CITES
RELATED_TO
```

---

## 4.2 Graph Retrieval Function

Graph 模块必须提供：

```python
def retrieve_graph(
    query: str,
    top_k: int = 5
) -> dict:
    ...
```

### Input Example

```python
query = "What is the relationship between VGAE and GCN?"
```

### Output Example

```json
{
  "entities": [
    "VGAE",
    "GCN"
  ],
  "triples": [
    {
      "source": "VGAE",
      "source_type": "Model",
      "relation": "USES",
      "target": "GCN",
      "target_type": "Model",
      "paper_id": "vgae_2016"
    }
  ],
  "paths": [],
  "paper_ids": [
    "vgae_2016"
  ]
}
```

### Required Top-level Fields

Graph Retriever 必须统一返回：

```text
entities
triples
paths
paper_ids
```

即使某个字段没有结果，也返回空列表。

例如：

```json
{
  "entities": [],
  "triples": [],
  "paths": [],
  "paper_ids": []
}
```

不要返回：

```python
None
```

---

## 4.3 Triple Schema

每个知识三元组统一使用：

```json
{
  "source": "VGAE",
  "source_type": "Model",
  "relation": "USES",
  "target": "GCN",
  "target_type": "Model",
  "paper_id": "vgae_2016"
}
```

### Required Fields

| Field | Type | Description |
|---|---|---|
| `source` | `string` | 起始实体 |
| `source_type` | `string` | 起始实体类型 |
| `relation` | `string` | 关系 |
| `target` | `string` | 目标实体 |
| `target_type` | `string` | 目标实体类型 |
| `paper_id` | `string` | 关系来源论文 |

---

## 4.4 Path Schema

第一阶段如果暂时没有实现多跳 Graph Retrieval，可以直接：

```json
{
  "paths": []
}
```

第二阶段实现 Multi-hop Retrieval 后，路径统一表示为：

```json
{
  "nodes": [
    "GCN",
    "VGAE",
    "Link Prediction"
  ],
  "relations": [
    "USED_BY",
    "SOLVES"
  ],
  "score": 0.92
}
```

例如该路径表示：

```text
GCN
 ↑ USES
VGAE
 ↓ SOLVES
Link Prediction
```

预留 `paths` 字段的目的，是让后续升级 Graph Retriever 时不需要修改 Backend API。

---

# 5. HybridRAG Interface

负责人：HybridRAG / Backend Integration

Hybrid 模块负责同时调用：

```text
Vector Retriever
+
Graph Retriever
```

然后统一组织两种 Retrieval 的结果。

---

## 5.1 Hybrid Retrieval Function

建议提供：

```python
def retrieve_hybrid(
    query: str,
    vector_top_k: int = 5,
    graph_top_k: int = 5
) -> dict:
    ...
```

### Output

```json
{
  "vector_results": [],
  "graph_results": {
    "entities": [],
    "triples": [],
    "paths": [],
    "paper_ids": []
  }
}
```

### Basic Implementation

```python
vector_results = retrieve_vector(
    query,
    top_k=vector_top_k
)

graph_results = retrieve_graph(
    query,
    top_k=graph_top_k
)

return {
    "vector_results": vector_results,
    "graph_results": graph_results
}
```

---

# 6. Prompt Builder Interface

Hybrid 模块负责将 Vector 和 Graph 的检索结果转换为 LLM 可以直接读取的 Context。

建议接口：

```python
def build_prompt(
    query: str,
    vector_results: list[dict],
    graph_results: dict,
    mode: str = "hybrid"
) -> str:
    ...
```

### Example

最终 Prompt 可以组织为：

```text
Question:
What is the relationship between VGAE and GCN?

Vector Evidence:

[Paper: vgae_2016]
The encoder of VGAE is based on a graph convolutional network.

Graph Evidence:

VGAE --USES--> GCN

Please answer the question using the provided evidence.
```

---

# 7. LLM Interface

LLM 模块与具体的大模型服务解耦。

建议统一提供：

```python
def generate_answer(
    prompt: str
) -> str:
    ...
```

具体模型可以替换，例如：

```text
DeepSeek
Qwen
OpenAI
```

在 VectorRAG、GraphRAG 和 HybridRAG 对比实验中，应尽量使用相同的生成模型，以保证实验公平。

---

# 8. Backend API Interface

负责人：Backend / HybridRAG Integration

Frontend 只通过 HTTP API 与 Backend 交互。

Frontend 不直接调用：

- Vector Database
- Neo4j
- Embedding Model
- LLM API

---

## 8.1 Ask API

Endpoint：

```http
POST /ask
```

Content-Type：

```text
application/json
```

### Request

```json
{
  "query": "VGAE 和 GCN 有什么关系？",
  "mode": "hybrid"
}
```

### Request Fields

| Field | Type | Description |
|---|---|---|
| `query` | `string` | 用户问题 |
| `mode` | `string` | 检索模式 |

`mode` 允许：

```text
vector
graph
hybrid
```

分别表示：

```text
vector → VectorRAG

graph → GraphRAG

hybrid → HybridRAG
```

---

## 8.2 Response

Backend 统一返回：

```json
{
  "query": "VGAE 和 GCN 有什么关系？",
  "mode": "hybrid",
  "answer": "VGAE uses a GCN-based encoder.",
  "vector_results": [
    {
      "paper_id": "vgae_2016",
      "chunk_id": "vgae_2016_003",
      "text": "The encoder of VGAE is based on a graph convolutional network...",
      "score": 0.91
    }
  ],
  "graph_results": {
    "entities": [
      "VGAE",
      "GCN"
    ],
    "triples": [
      {
        "source": "VGAE",
        "source_type": "Model",
        "relation": "USES",
        "target": "GCN",
        "target_type": "Model",
        "paper_id": "vgae_2016"
      }
    ],
    "paths": [],
    "paper_ids": [
      "vgae_2016"
    ]
  },
  "references": [
    {
      "paper_id": "vgae_2016",
      "title": "Variational Graph Auto-Encoders"
    }
  ]
}
```

---

# 9. Frontend Interface

负责人：Frontend

Frontend 不需要了解 Backend 内部使用：

```text
FAISS
Chroma
Neo4j
Embedding
Cypher
Graph Retrieval
LLM Prompt
```

Frontend 只负责：

```text
用户输入问题
      ↓
POST /ask
      ↓
Backend 返回 JSON
      ↓
解析 Response
      ↓
展示结果
```

Frontend 至少需要展示：

```text
Question

Retrieval Mode
- Vector
- Graph
- Hybrid

Final Answer

Vector Evidence

Graph Evidence

References
```

---

# 10. Error Handling

如果 Vector Retriever 没有检索到结果：

```json
{
  "vector_results": []
}
```

如果 Graph Retriever 没有检索到结果：

```json
{
  "graph_results": {
    "entities": [],
    "triples": [],
    "paths": [],
    "paper_ids": []
  }
}
```

HybridRAG 不应该因为其中一个 Retriever 没有结果而直接崩溃。

例如：

```text
Vector 有结果 + Graph 无结果
→ 可以继续使用 Vector Context 生成答案

Vector 无结果 + Graph 有结果
→ 可以继续使用 Graph Context 生成答案

两者均无结果
→ 返回无法找到相关证据的信息
```

---

# 11. Module Dependency

```text
A Data Processing
       ↓
papers.jsonl
       ↓
 ┌─────┴─────┐
 ↓           ↓
B Vector    C Graph
Retriever   Retriever
 └─────┬─────┘
       ↓
HybridRAG
       ↓
Prompt Builder
       ↓
LLM
       ↓
Backend API
       ↓
D Frontend
```

模块之间只依赖本文档规定的接口，不依赖其他模块内部具体实现。

---

# 12. Shared Identifier Rules

所有模块必须统一使用 `paper_id`。

例如 VGAE：

```text
vgae_2016
```

A 输出：

```json
{
  "paper_id": "vgae_2016"
}
```

B 的 Chunk：

```json
{
  "paper_id": "vgae_2016",
  "chunk_id": "vgae_2016_003"
}
```

C 的 Triple：

```json
{
  "paper_id": "vgae_2016",
  "source": "VGAE",
  "relation": "USES",
  "target": "GCN"
}
```

禁止同一篇论文在不同模块中使用不同 ID，例如：

```text
VGAE
variational_graph_autoencoder
vgae-paper
vgae_2016
```

应统一为：

```text
vgae_2016
```

---

# 13. Interface Freeze

第一阶段开发期间，以下接口视为稳定接口。

## Vector Retriever

```python
retrieve_vector(
    query: str,
    top_k: int = 5
) -> list[dict]
```

## Graph Retriever

```python
retrieve_graph(
    query: str,
    top_k: int = 5
) -> dict
```

## LLM

```python
generate_answer(
    prompt: str
) -> str
```

## Backend

```http
POST /ask
```

如果需要修改：

- 函数名
- 参数
- 返回值
- JSON 字段
- `paper_id` 规则

必须提前通知所有受到影响的成员。

---

# 14. Responsibilities

| Member | Module | Main Output |
|---|---|---|
| A | Data Processing | `papers.jsonl` + 数据处理 Pipeline |
| B | Vector RAG | Vector Index + `retrieve_vector()` |
| C | Knowledge Graph / Graph Retrieval | KG + `retrieve_graph()` |
| D | Frontend | 用户界面 + `/ask` API 调用 |
| Leader | HybridRAG / Backend | Hybrid Fusion + Prompt + LLM + Backend Integration |

---

# 15. Development Principle

项目开发遵循以下原则：

```text
内部实现可以不同
        ↓
对外接口必须统一
        ↓
各模块可以独立开发
        ↓
最终可以直接集成
```

各成员首先确保自己的模块满足接口要求，再进行算法优化。

第一阶段优先目标：

```text
模块能运行
+
接口能连接
+
系统能形成完整闭环
```

第二阶段再重点优化：

```text
Graph Retrieval
HybridRAG
Multi-hop Retrieval
Query-aware Retrieval
Evaluation
Explainability
```