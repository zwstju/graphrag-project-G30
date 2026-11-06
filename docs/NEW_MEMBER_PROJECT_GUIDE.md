# GraphRAG 项目新成员说明

## 1. 项目简介

本项目是一个面向图神经网络（GNN）科研论文的检索增强问答系统。

用户可以提出类似下面的问题：

> 哪个模型使用 GCN 作为编码器？

系统不会只依赖大模型已有知识，而是先从本地论文语料中检索证据，再要求大模型依据证据回答，并标注论文来源。

项目的核心研究问题是：

> 如何从文本检索和知识图谱检索产生的异构候选中，选出真正能够支持答案的科研证据？

因此，本项目的重点不仅是把 VectorRAG 和 GraphRAG 放在一起，更重要的是统一、重排和验证不同类型的证据。

---

## 2. 需要先理解的概念

### VectorRAG

VectorRAG 将论文文本编码成向量，通过语义相似度检索与问题最接近的文本片段。

优点是覆盖面广，适合事实和描述类问题。缺点是“语义相似”不一定代表“逻辑上能够回答问题”。

### GraphRAG

GraphRAG 将论文知识表示成实体关系三元组，例如：

```text
VGAE --USES--> GCN
GraphSAGE --EVALUATED_ON--> Reddit
SGC --SIMPLIFIES--> GCN
```

它适合关系和多跳问题，但效果依赖知识图谱是否完整、实体是否正确匹配。

### HybridRAG

HybridRAG 同时使用 Vector 文本证据和 Graph 三元组/路径证据，以提高证据覆盖率。

直接拼接两类证据可能引入噪声，因此 Stage 2 增加了 Evidence Reranker，对候选证据统一打分和筛选。

---

## 3. 项目总体流程

项目分为离线构建和在线问答两部分。

### 离线构建

```text
原始 PDF
    ↓
PDF 文本提取和清洗
    ↓
data/processed/papers.jsonl
    ↓
┌──────────────────────┐
│                      │
▼                      ▼
结构感知文本分块       LLM 三元组抽取
│                      │
▼                      ▼
BGE-M3 向量编码        实体规范化与去重
│                      │
▼                      ▼
FAISS 向量索引         data/graph/triples.jsonl
```

### 在线问答

```text
用户问题
    ↓
┌──────────────────────┐
│                      │
▼                      ▼
Vector 检索            Graph 检索
文本块                  三元组和路径
│                      │
└──────────┬───────────┘
           ▼
      统一 Evidence
           ↓
     Evidence Reranker
           ↓
       Top-6 证据
           ↓
    大模型生成带引用答案
           ↓
  引用检查和 Claim 支持验证
```

---

## 4. Stage 1 和 Stage 2 的区别

仓库中同时保留了两代实现。

### Stage 1：基础系统

代码位于 `src/`。

它负责跑通最初的完整闭环：

```text
VectorRAG / GraphRAG / 简单 HybridRAG / API / 前端
```

Stage 1 的 Hybrid 方法主要是直接拼接 Vector 和 Graph 上下文。

### Stage 2：最终实验系统

代码位于 `stage2/`。

它在 Stage 1 基础上增加了：

- 结构感知分块；
- 持久化且带完整性校验的向量索引；
- 更严格的图检索；
- 统一 Evidence 数据结构；
- 可解释的证据重排；
- 引用检查和 Claim 验证；
- 受控消融实验。

最终实验和项目汇报应以 Stage 2 为主。Stage 2 复用 Stage 1 构建的 `data/graph/triples.jsonl`，不会重新抽取另一套知识图谱。

---

## 5. 目录结构

```text
graphrag-project-G30/
├── configs/          Stage 2 和实验配置
├── data/             PDF、论文、分块、图谱和索引
├── benchmark/        Core80、Core69、Core32 问题集
├── src/              Stage 1 基础系统和知识图谱构建
├── stage2/           最终 Evidence-Centric 系统
├── scripts/          构建、审计、实验和结果整理入口
├── tests/            自动化测试
├── frontend/         Stage 1 前端
├── frontend_stage2/  Stage 2 前端
├── results/          实验原始结果、表格和图
└── docs/             项目文档
```

---

## 6. 数据和配置

### 核心论文语料

```text
data/processed/papers.jsonl
```

每行代表一篇论文，主要字段包括：

```json
{
  "paper_id": "vgae_2016",
  "title": "Variational Graph Auto-Encoders",
  "authors": ["Thomas Kipf", "Max Welling"],
  "year": 2016,
  "text": "论文正文"
}
```

`paper_id` 是贯穿全部模块的来源标识。Vector Evidence、Graph Triple、Benchmark 和答案引用必须使用一致的 `paper_id`。

### Stage 2 主要配置

```text
configs/default.json
```

其中配置了：

- 论文、图谱、分块和索引路径；
- BGE-M3 嵌入模型；
- Vector 和 Graph 候选数量；
- 最终证据数量；
- Graph 最大跳数；
- Reranker 权重。

`configs/fixed_chunk.json` 是固定字符分块对照组的配置。

---

## 7. Stage 1 模块说明

### `src/data_processing/`

`process_papers.py` 负责从 PDF 提取和清洗文本，输出 `papers.jsonl`。

### `src/vector/`

`retriever.py` 是早期 VectorRAG 实现，包含固定字符分块、BGE-M3 编码、FAISS 建索引和查询。

该版本主要作为 Stage 1 基线，最终实验使用 `stage2/retrieval/vector.py`。

### `src/graph/`

图模块分为“建图”和“查图”两部分。

#### 建图相关文件

| 文件 | 作用 |
|---|---|
| `schema.py` | 定义允许的实体类型和关系类型 |
| `utils.py` | 文本规范化、模型 JSON 解析和 JSONL 读写 |
| `chunking.py` | 将论文切成供图谱抽取使用的大文本块 |
| `llm.py` | 调用项目配置的大模型接口 |
| `extractor.py` | 精炼论文文本并抽取实体关系三元组 |
| `entity_resolution.py` | 统一实体别名并删除重复三元组 |
| `build_graph.py` | 串联上述步骤并写出图谱文件 |

建图过程是：

```text
papers.jsonl
→ 图谱抽取分块
→ LLM 精炼文本
→ LLM 抽取三元组
→ 实体规范化
→ 三元组去重
→ data/graph/triples.jsonl
```

#### 查图相关文件

| 文件 | 作用 |
|---|---|
| `query_parser.py` | 从用户问题中识别知识图谱种子实体 |
| `retriever.py` | 加载图谱，检索相关三元组和实体路径 |

Stage 1 查询实体时优先使用字面匹配；完全匹配不到时，可以调用 LLM 从已有图节点中选择实体。LLM 不允许创造图中不存在的新节点。

### `src/query/`

`processor.py` 负责基础的查询实体识别和意图分类，主要服务于 Stage 1 Prompt。

### `src/retrieval/`

`adapters.py` 为 Vector 和 Graph 检索器提供统一调用接口，并支持开发阶段的 mock 数据。

### `src/hybrid/`

- `pipeline.py`：根据 `vector`、`graph`、`hybrid` 模式组织完整问答流程；
- `prompt.py`：把文本块、三元组和路径格式化为 LLM 上下文。

### `src/llm/`

`client.py` 负责 Stage 1 最终答案生成。它与 `src/graph/llm.py` 的职责不同：前者生成答案，后者主要服务于知识图谱构建和实体链接。

### `src/backend/`

- `app.py`：Stage 1 FastAPI 后端；
- `app_stage2.py`：Stage 2 演示后端。

---

## 8. Stage 2 模块说明

### `stage2/schema.py`

定义统一的 `Evidence` 数据结构。

无论证据来自 Vector、Triple 还是 Path，都转换成类似下面的对象：

```python
Evidence(
    evidence_id="V1",
    source_type="vector",
    content="证据内容",
    paper_id="vgae_2016",
    section="Method",
    retrieval_score=0.82,
    rerank_score=0.91,
)
```

### `stage2/data/`

- `structure_chunker.py`：识别 Abstract、Method、Experiments 等章节，并在章节内部按段落和句子分块；
- `fixed_chunker.py`：固定字符分块对照组，用于消融实验。

结构化分块首先生成 `structure_chunks.jsonl`，随后排除 References，得到真正用于检索的 `retrieval_chunks.jsonl`。

### `stage2/retrieval/`

#### `vector.py`

负责：

- 使用 BGE-M3 生成归一化向量；
- 使用 FAISS `IndexFlatIP` 建立索引；
- 保存索引、元数据和完整性 Manifest；
- 在线编码问题并检索文本证据。

#### `graph.py`

负责：

- 加载 `data/graph/triples.jsonl`；
- 使用边界感知方式匹配查询实体；
- 过滤 `CITES`、`WRITTEN_BY` 等普通 QA 噪声关系；
- 检索相关三元组；
- 查找不超过配置跳数的图路径；
- 保留证据涉及的全部论文来源。

### `stage2/rerank/`

`evidence_reranker.py` 将 Vector、Triple 和 Path 放入同一个候选池，根据以下特征统一打分：

- 语义相似度；
- 实体重叠；
- 关系匹配；
- 论文章节适配；
- 证据来源类型；
- 原始检索分数。

当前实现是可解释的规则 Reranker，不是 Cross-Encoder。各部分分数会保存在 Evidence 元数据中。

### `stage2/llm/`

- `client.py`：统一的大模型 API 客户端；
- `generate.py`：将问题和最终 Evidence 构造成回答 Prompt。

大模型被要求只能依据给定证据回答，并在关键事实后使用 `[paper_id]` 引用。

### `stage2/verify/`

`citation.py` 包含两种检查：

1. 确定性检查引用的 `paper_id` 是否来自当前 Evidence；
2. 将答案确定性拆成 Claims，再让 LLM 逐条判断 Evidence 是否直接支持 Claim。

Claim 验证是生成后的评估步骤，不会重新生成或修改答案。

### `stage2/eval/`

- `metrics.py`：Token F1、论文级检索召回率代理指标和证据精确率代理指标；
- `judge.py`：使用统一 LLM Judge 评价答案正确性和切题程度。

### `stage2/pipeline.py`

这是 Stage 2 的总入口，将 Vector、Graph、Reranker、生成和验证模块连接起来。

支持五种模式：

| 模式 | 含义 |
|---|---|
| `vector` | 只使用 Vector 证据 |
| `graph` | 只使用 Graph 证据 |
| `hybrid` | Vector 和 Graph 按固定配额直接组合 |
| `rerank` | Vector 与 Graph 形成候选池并统一重排 |
| `full` | 使用相同的 Rerank 结果生成答案，再执行验证 |

`full` 不会因为增加验证而自动改善答案，它只是增加了 post-hoc 检查。

---

## 9. Benchmark 与实验

Benchmark 文件位于 `benchmark/`。

- `gnn_qa_core80.jsonl`：80 个候选问题；
- `gnn_qa_core80_audit_smoke8.jsonl`：审计流程的 8 题 Smoke Test；
- `gnn_qa_core69_audit_filtered.jsonl`：通过 LLM Evidence Audit 的自动评测集合；
- `gnn_qa_core32_generation_stratified.jsonl`：用于生成评测的分层 32 题子集。

问题类型包括：

```text
Fact / Relation / Comparison / Multi-hop
```

Core80 首先经过“候选答案证据审计”：系统从每篇声明的 Gold Paper 中独立选择证据，再由 LLM 判断候选答案是否得到这些论文证据支持。

这个过程不是普通检索评测，也不等同于人工验证。

---

## 10. `scripts/` 的组织方式

`stage2/` 提供可复用功能，`scripts/` 负责组织一次完整任务并保存实验 Artifact。

### 数据和索引构建

```text
build_structure_chunks.py
build_retrieval_chunks.py
build_fixed_chunks.py
build_vector_index.py
```

### Benchmark 管理和审计

```text
validate_benchmark.py
build_core80_audit_smoke.py
audit_benchmark_llm.py
summarize_benchmark_audit.py
freeze_core69.py
build_generation_subset.py
```

### 实验执行

```text
run_retrieval_ablation.py
run_generation_evaluation.py
run_citation_verification.py
```

### 结果处理

```text
aggregate_final_results.py
plot_results.py
build_case_studies.py
```

---

## 11. 一个完整查询示例

问题：

> Which model uses GCN as an encoder?

执行过程：

1. BGE-M3 将问题编码成查询向量；
2. FAISS 找到与 GCN、encoder、VGAE 相关的论文文本；
3. Graph Retriever 从问题中识别种子实体 `GCN`；
4. 图谱返回 `VGAE --USES--> GCN`；
5. 文本块和三元组转换为统一 Evidence；
6. Reranker 根据语义、实体和关系匹配对证据排序；
7. 最终 Top-6 Evidence 被交给大模型；
8. 大模型根据证据生成带 `[vgae_2016]` 引用的答案；
9. 程序检查引用是否来自当前证据；
10. LLM Verifier 逐条判断答案 Claim 是否被 Evidence 支持。

---

## 12. 哪些步骤使用大模型

使用生成式大模型的环节：

- 图谱构建前的文本精炼；
- 实体关系三元组抽取；
- Stage 1 实体链接 fallback；
- 最终答案生成；
- Benchmark Evidence Audit；
- 答案正确性 Judge；
- Claim 支持判断。

不使用生成式大模型的环节：

- PDF 清理；
- 章节识别和文本分块；
- BGE-M3 向量编码；
- FAISS 搜索；
- Stage 2 图实体匹配和路径搜索；
- 规则 Reranker；
- 引用 ID 检查；
- Token F1 和论文级检索指标。

BGE-M3 是嵌入模型，不是负责生成答案的聊天大模型。

---

## 13. 当前系统的边界

- 知识图谱由 LLM 自动抽取，可能遗漏信息；
- 实体规范化主要依赖规则和手工别名；
- Reranker 是规则基线，不是学习型 Cross-Encoder；
- 检索 Recall/Precision 是论文级代理指标，不是段落级人工相关性标注；
- LLM Claim Verifier 是评估工具，不应被视为绝对事实；
- 验证目前不会反馈给生成模型自动改写答案；
- Vector 检索在复杂比较和关系推理问题上仍可能返回语义相关但逻辑无关的证据。

---

## 14. 推荐阅读顺序

第一次阅读项目时，建议按照下面的顺序：

```text
configs/default.json
→ stage2/schema.py
→ src/graph/schema.py
→ src/graph/utils.py
→ src/graph/chunking.py
→ src/graph/llm.py
→ src/graph/extractor.py
→ src/graph/entity_resolution.py
→ src/graph/build_graph.py
→ src/graph/query_parser.py
→ stage2/retrieval/vector.py
→ stage2/retrieval/graph.py
→ stage2/rerank/evidence_reranker.py
→ stage2/llm/generate.py
→ stage2/verify/citation.py
→ stage2/pipeline.py
```

图模块内部可以记成：

```text
schema → 规定格式
utils → 公共工具
chunking → 切分论文
llm → 调用模型
extractor → 抽取三元组
entity_resolution → 统一实体
build_graph → 组织建图
query_parser → 找查询实体
retriever → 检索图证据
```

---

## 15. 一句话总结

本项目先用 Vector 和 Graph 两条检索路径扩大科研证据覆盖范围，再通过统一 Evidence 和可解释 Reranker 控制证据质量，最后让大模型基于筛选后的证据生成带论文引用的答案，并使用引用检查和 Claim 验证评估答案是否可追溯。
