# 组长：HybridRAG、系统整合与实验设计说明

## 1. 角色定位

组长负责把 Data、Vector、Graph 和 Frontend 四部分连接成一个能够运行、评测和展示的完整系统。

这个角色不需要替代其他成员实现各自的核心模块，主要职责是：

- 设计整体架构和模块接口；
- 接入 Vector 与 Graph 检索结果；
- 实现 HybridRAG Pipeline；
- 统一不同来源的 Evidence；
- 构造回答 Prompt 并调用 LLM；
- 提供后端 API；
- 设计公平的对照实验；
- 聚合结果并统筹阶段汇报和答辩。

一句话概括：

> 其他成员分别生产数据、Vector 证据、Graph 证据和展示界面，组长负责把它们组织成可验证的端到端系统。

---

## 2. 主要负责的代码

### Stage 1 系统整合

| 文件 | 作用 |
|---|---|
| `src/query/processor.py` | 对用户问题做基础实体识别和意图分类 |
| `src/retrieval/adapters.py` | 统一调用 B 的 Vector 检索器和 C 的 Graph 检索器 |
| `src/hybrid/prompt.py` | 将文本块、三元组和路径组织成 LLM Prompt |
| `src/hybrid/pipeline.py` | 根据 Vector、Graph、Hybrid 模式执行完整问答流程 |
| `src/llm/client.py` | 调用回答生成模型，并支持开发阶段 Mock |
| `src/backend/app.py` | 提供 Stage 1 FastAPI `/ask` 接口 |

Stage 1 Hybrid 的核心形式是：

```text
Vector Context
+
Graph Context
→ Prompt
→ LLM Answer
```

这一版本主要证明各模块能够按照统一接口协作。

### Stage 2 系统整合

| 文件 | 作用 |
|---|---|
| `stage2/schema.py` | 定义统一 Evidence 和 QAItem 数据结构 |
| `stage2/pipeline.py` | 连接 Vector、Graph、Reranker、生成和验证模块 |
| `stage2/rerank/evidence_reranker.py` | 对异构 Evidence 统一打分并选出最终证据 |
| `stage2/llm/client.py` | 为生成、Judge 和验证提供统一 LLM 客户端 |
| `stage2/llm/generate.py` | 构造证据约束 Prompt 并生成带引用答案 |
| `stage2/verify/citation.py` | 检查引用并进行 Claim-level Evidence Verification |
| `stage2/eval/metrics.py` | 计算规则型检索和答案指标 |
| `stage2/eval/judge.py` | 使用统一 LLM Judge 评价答案正确性与相关性 |
| `src/backend/app_stage2.py` | 提供 Stage 2 `/api/ask` 和前端数据接口 |

### 实验和结果组织

| 文件 | 作用 |
|---|---|
| `scripts/audit_benchmark_llm.py` | 对候选 Benchmark 答案做 Gold-paper-aware Evidence Audit |
| `scripts/freeze_core69.py` | 冻结通过自动证据审计的评测集合 |
| `scripts/build_generation_subset.py` | 构建分层 Core32 生成评测集 |
| `scripts/run_retrieval_ablation.py` | 执行受控检索消融实验 |
| `scripts/run_generation_evaluation.py` | 使用保存证据生成答案并执行统一 Judge |
| `scripts/run_citation_verification.py` | 对保存的原始答案做 Claim 验证 |
| `scripts/aggregate_final_results.py` | 聚合最终表格和指标 |
| `scripts/build_case_studies.py` | 整理成功与失败案例 |

主要配置由下面两个文件管理：

```text
configs/default.json
configs/fixed_chunk.json
```

---

## 3. 与其他成员的接口

### 从 A：Data 接收

```text
data/processed/papers.jsonl
```

要求：

- `paper_id` 唯一且稳定；
- title、authors、year、text 等字段完整；
- PDF 提取质量经过检查；
- 不把未成功处理的论文当成已进入语料。

### 从 B：Vector 接收

Stage 1 接口：

```python
retrieve_vector(query: str, top_k: int) -> list[dict]
```

Stage 2 接口：

```python
VectorIndex.search(query: str, top_k: int) -> list[Evidence]
```

### 从 C：Graph 接收

Stage 1 接口：

```python
retrieve_graph(query: str, top_k: int) -> dict
```

Stage 2 接口：

```python
GraphRetriever.search(query: str, top_k: int) -> list[Evidence]
```

### 向 D：Frontend 提供

Stage 2 后端主要返回：

```text
query
method
answer
evidence
before_evidence
graph
checks
timings
request_latency_ms
```

D 只依赖这个返回契约，不需要了解 FAISS 或 NetworkX 的内部实现。

---

## 4. 组长负责的核心执行流程

```text
用户问题
   ↓
Stage2System.retrieve()
   ↓
Vector 候选 + Graph 候选
   ↓
统一 Evidence Pool
   ↓
EvidenceReranker.rerank()
   ↓
Top-6 Evidence
   ↓
answer_question()
   ↓
带 [paper_id] 引用的答案
   ↓
引用合法性检查
   ↓
可选 Claim-level LLM Verification
```

Stage 2 支持五种模式：

| 模式 | 用途 |
|---|---|
| `vector` | 纯 VectorRAG 基线 |
| `graph` | 纯 GraphRAG 基线 |
| `hybrid` | Vector 与 Graph 直接组合基线 |
| `rerank` | 统一候选池经过 Evidence Reranker |
| `full` | 使用与 rerank 相同的证据，并增加生成后验证 |

需要特别说明：`full` 的验证是 post-hoc 检查，不会重新生成答案，因此不能把它描述成自动纠错系统。

---

## 5. Reranker 应该怎样讲

Reranker 的输入是 B 和 C 返回的异构证据：

```text
Vector 文本块
Graph 三元组
Graph 路径
```

这些证据统一成 `Evidence` 后，根据以下信号打分：

```text
semantic  语义相关性
entity    实体重叠
relation  关系与查询意图的匹配程度
section   论文章节是否适合当前题型
source    Vector、Triple、Path 来源先验
retrieval 原始检索分数
```

配置中的当前权重为：

```text
0.40 × semantic
0.18 × entity
0.16 × relation
0.08 × section
0.08 × source
0.10 × retrieval
```

它是透明的规则基线，不是 Cross-Encoder。优势是每部分分数都可以解释，局限是规则不一定能够覆盖所有表达方式。

---

## 6. 实验设计应该怎样讲

项目不能只展示一个最终系统，需要通过受控对比说明每个改动的作用。

检索消融包含：

```text
Fixed Vector
Structured Vector
Structured Vanilla Hybrid
Structured Hybrid + Rerank
```

分别用于观察：

1. 结构感知分块相对固定分块的作用；
2. 加入 Graph Evidence 的作用；
3. 对相同 Vector+Graph 候选进行 Rerank 的作用。

生成评测使用同一组保存的 Evidence，比较：

```text
Structured Vector
Structured Vanilla Hybrid
Structured Hybrid + Rerank
```

主要报告：

- Paper-level retrieval recall proxy；
- Paper-level evidence precision proxy；
- Token F1；
- LLM correctness/relevance score；
- Citation validity；
- Claim support rate；
- 检索、生成和验证延迟；
- API 重试与失败数量。

基础设施失败必须与语义错误分开统计。

---

## 7. 建议讲解顺序

组长汇报时建议按照下面的顺序：

1. 介绍问题：科研论文问答中的检索证据可能语义相关但逻辑无关；
2. 解释为什么同时需要 Vector 和 Graph；
3. 展示整体离线构建与在线问答架构；
4. 区分 Stage 1 直接拼接和 Stage 2 Evidence-Centric 设计；
5. 介绍统一 Evidence 与 Reranker；
6. 介绍 Grounded Generation 和引用约束；
7. 介绍 Benchmark Audit、检索消融和生成评测；
8. 展示成功案例和失败案例；
9. 最后说明当前限制和后续方向。

### 可直接使用的简短讲解

> 我负责整个系统的架构与整合。A 交付标准论文语料，B 和 C 分别返回文本证据与图证据，我将它们统一成 Evidence。Stage 1 采用直接上下文拼接，Stage 2 则先形成统一候选池，再根据语义、实体、关系、章节、来源和原始检索分数进行可解释重排。最终证据交给大模型生成带论文引用的答案，再进行引用和 Claim 支持验证。我同时负责设计受控实验，确保结构化分块、Graph Evidence 和 Reranking 的影响能够分别比较。

---

## 8. 容易说错的地方

- 不要把 Vector 和 Graph 的直接拼接称为主要创新；
- 不要把 LLM Evidence Audit 称为人工验证；
- 不要把 Paper-level proxy 指标称为段落级证据准确率；
- 不要声称 `full` 会自动修正答案；
- 不要把 LLM Claim Verifier 当成绝对正确的事实标注；
- 不要把规则 Reranker 说成训练得到的神经网络模型；
- 不要只展示正面案例，也要报告仍然存在的失败模式。

---

## 9. 组长交付清单

- 模块接口能够连接；
- Stage 1 和 Stage 2 流程边界明确；
- Prompt 和 LLM 调用配置统一；
- Backend 返回字段稳定；
- 实验条件使用相同数据、问题和证据预算；
- 原始请求级结果得到保留；
- 结果表格和图只使用真实测量数据；
- 汇报中正确区分候选答案、LLM Audit 和人工验证。
