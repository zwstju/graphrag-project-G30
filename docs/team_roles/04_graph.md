# C：Knowledge Graph 与 GraphRAG 模块说明

## 1. 角色定位

Graph 模块负责从论文中抽取结构化知识，完成实体规范化，并根据问题检索相关三元组和多跳路径。

一句话概括：

> Graph 模块把论文中的“谁使用了什么、谁提出了什么、模型在哪个数据集评估”等关系变成可查询的图证据。

Vector 检索擅长寻找语义相似文本，Graph 检索更强调明确的实体关系和连接路径。

---

## 2. 主要负责的代码和文件

```text
src/graph/
├── schema.py
├── utils.py
├── chunking.py
├── llm.py
├── extractor.py
├── entity_resolution.py
├── build_graph.py
├── query_parser.py
└── retriever.py
```

Stage 2 还包括：

```text
stage2/retrieval/graph.py
tests/test_stage2_graph_reranker.py
```

图数据文件：

```text
data/graph/triples_raw.jsonl
data/graph/triples.jsonl
```

---

## 3. 图模块分成两个阶段

### 离线：从论文构建知识图谱

```text
papers.jsonl
→ 图谱抽取分块
→ LLM 精炼文本
→ LLM 抽取实体和关系
→ 实体规范化
→ 重复三元组删除
→ triples.jsonl
```

### 在线：根据问题检索图证据

```text
用户问题
→ 查询实体匹配
→ Seed Nodes
→ 相关三元组打分
→ 最短路径搜索
→ Triple / Path Evidence
```

---

## 4. 每个文件的职责

### `schema.py`

定义允许的大类实体：

```text
Paper
Model
Author
Dataset
Task
Method
Concept
Year
```

定义允许的关系：

```text
PROPOSES
USES
EVALUATED_ON
SOLVES
WRITTEN_BY
CITES
EXTENDS
SIMPLIFIES
RELATED_TO
```

这是 LLM 三元组抽取的结构约束，可以减少关系名称随意变化。

### `utils.py`

提供图模块共用功能：

- 规范化文本和实体比较键；
- 删除模型输出中的 Markdown Fence；
- 从模型混合输出中提取 JSON 数组；
- 读取和写入 JSONL。

### `chunking.py`

将论文正文切成供知识图谱抽取使用的大文本块。

默认配置：

```text
chunk_size = 6000
overlap = 500
```

这个分块服务于 LLM 图谱抽取，不是 Vector 检索分块。

### `llm.py`

读取 `.env` 中的模型配置，调用兼容 OpenAI Chat Completions 的接口。

主要服务于：

- 论文块精炼；
- 实体和关系抽取；
- Stage 1 查询实体链接的 LLM fallback。

### `extractor.py`

包含两层 LLM 处理。

第一层 `refine_chunk()`：

- 保留模型、数据集、任务和方法；
- 保留显式技术关系；
- 删除排版和重复噪声；
- 禁止增加论文没有的事实。

第二层 `extract_triples()`：

将文本转换成：

```json
{
  "source": "VGAE",
  "source_type": "Model",
  "relation": "USES",
  "target": "GCN",
  "target_type": "Model",
  "paper_id": "vgae_2016",
  "chunk_id": "vgae_2016_003"
}
```

程序会过滤不在 Schema 中的实体类型和关系。

### `entity_resolution.py`

统一同一实体的不同写法，例如：

```text
Graph Convolutional Network
Graph Convolutional Networks
GCN
```

统一为 `GCN`，然后按照实体、关系和论文来源删除重复三元组。

当前实现主要依赖文本规范化和手工别名，不是学习型实体消歧系统。

### `build_graph.py`

知识图谱构建总入口，负责串联分块、LLM 抽取、实体规范化和文件写出。

先写出未经实体规范化的：

```text
triples_raw.jsonl
```

再写出用于正式检索的：

```text
triples.jsonl
```

### `query_parser.py`

Stage 1 查询实体识别：

1. 优先用规范化后的字面匹配；
2. 长实体优先，避免短实体子串误匹配；
3. 如果没有任何字面结果，再调用 LLM；
4. LLM 只能从已有图节点中选择，不能创造新节点。

### `retriever.py`

Stage 1 图检索器：

- 将三元组加载到 NetworkX `MultiDiGraph`；
- 以查询实体作为 Seed；
- 执行固定一跳扩展；
- 对候选三元组进行规则打分；
- 查找 Seed 之间的最短路径；
- 返回实体、三元组、路径和论文来源。

### `stage2/retrieval/graph.py`

Stage 2 复用同一份 `triples.jsonl`，但改进了在线检索：

- 使用实体边界匹配，避免 `uses GCN` 错误匹配成 `SGC`；
- 排除 `CITES` 和 `WRITTEN_BY` 等普通 QA 噪声关系；
- 根据查询关系意图对边进行加分；
- 对 MultiDiGraph 平行边进行确定性选择；
- 支持配置范围内的最短路径；
- 路径保留沿途所有 `paper_id`；
- 严格控制返回 `top_k`；
- 返回统一 `Evidence` 对象。

---

## 5. 为什么使用 MultiDiGraph

两个实体之间可能同时存在不同关系，例如：

```text
Model A --USES--> Method B
Model A --EXTENDS--> Method B
```

普通有向图可能覆盖其中一条边，`MultiDiGraph` 可以保存多个平行关系。

检索时再根据查询语义确定当前更适合使用哪条关系。

---

## 6. Triple 和 Path Evidence

### Triple Evidence

表示单条明确关系：

```text
VGAE --USES--> GCN
```

适合回答直接关系问题。

### Path Evidence

表示多个实体之间的连接：

```text
Model A --USES--> Method B --EVALUATED_ON--> Dataset C
```

适合回答需要组合两条或多条关系的问题。

Path 不能被当成无限制推理。系统只返回图中实际存在、且不超过最大跳数的路径。

---

## 7. 与其他成员的接口

### 从 A：Data 接收

```text
data/processed/papers.jsonl
```

Graph 模块需要保留每条三元组对应的 `paper_id` 和 `chunk_id`，否则最终证据无法追溯。

### 向组长交付

Stage 1：

```python
retrieve_graph(query, top_k) -> {
    "entities": [],
    "triples": [],
    "paths": [],
    "paper_ids": []
}
```

Stage 2：

```python
GraphRetriever.search(query, top_k) -> list[Evidence]
```

Evidence 的 `source_type` 是：

```text
triple
path
```

### 与 B：Vector 的关系

Graph 不替代 Vector：

- Vector 提供完整文本上下文；
- Graph 提供显式关系和路径；
- 两类证据最终由组长的 Reranker 统一选择。

### 与 D：Frontend 的关系

D 使用 Graph Evidence 中的：

```text
source
relation
target
nodes
relations
paper_ids
```

绘制图节点、边和关系路径。

---

## 8. 建议讲解顺序

1. 用 `VGAE --USES--> GCN` 解释三元组；
2. 打开 `schema.py`，说明实体和关系约束；
3. 打开 `chunking.py` 和 `llm.py`，说明怎样准备 LLM 输入；
4. 打开 `extractor.py`，说明两层 LLM 抽取；
5. 打开 `entity_resolution.py`，说明别名统一与去重；
6. 打开 `build_graph.py`，串起完整离线建图；
7. 打开 `query_parser.py`，说明怎样从问题找到 Seed；
8. 打开 Stage 2 Graph Retriever，说明 Triple、Path、降噪和 Top-k；
9. 展示一条真实 Triple 和一条 Path Evidence。

### 可直接使用的简短讲解

> 我负责 Knowledge Graph 和 GraphRAG。首先把 A 提供的论文正文切成适合抽取的大文本块，通过两层 LLM 先精炼文本，再按照限定 Schema 抽取实体关系三元组。之后对 GCN、VGAE 等实体进行别名规范化并删除重复三元组。在线查询时，从问题中识别图节点，检索与种子实体相关的三元组，并在多实体问题中查找有限跳数的路径。Stage 2 还加入了噪声关系过滤、边界感知实体匹配、平行边选择和完整论文来源记录，最后将 Triple 和 Path 作为统一 Evidence 交给 Hybrid Pipeline。

---

## 9. 演示时适合展示什么

- `schema.py` 中允许的关系类型；
- `triples_raw.jsonl` 和 `triples.jsonl` 的区别；
- `Graph Convolutional Network` 被规范化为 `GCN`；
- `VGAE --USES--> GCN` 的直接关系证据；
- 一条包含多个论文来源的 Path Evidence；
- `CITES` 被过滤前后的噪声差异。

---

## 10. 容易说错的地方

- 不要说图谱是人工逐条标注的，它主要由 LLM 自动抽取；
- 不要说所有关系都经过人工确认；
- 不要把一条图路径自动等同于严格逻辑证明；
- 不要把 Stage 1 固定一跳和 Stage 2 最大三跳混淆；
- 不要说实体规范化使用了训练模型，当前主要是规则和手工别名；
- 不要忽略三元组和路径的论文来源；
- 不要声称 Graph 能覆盖论文中的全部文本事实。

---

## 11. Graph 交付清单

- 抽取结果符合实体和关系 Schema；
- 每条三元组保留 `paper_id` 和 `chunk_id`；
- 实体别名得到统一；
- 重复三元组得到处理；
- 图检索不会返回普通 QA 中明显无效的噪声关系；
- `top_k` 得到严格执行；
- 多跳路径保留全部论文来源；
- 输出格式符合组长定义的接口。
