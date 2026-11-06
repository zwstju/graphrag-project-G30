# D：Frontend、证据展示与协作说明

## 1. 角色定位

Frontend 模块负责把后端能力转换成用户可以理解和操作的界面，并协助整理文档、图表和案例。

一句话概括：

> 前端不只展示最终答案，还需要展示答案来自哪些论文、使用了哪些文本块或图关系，以及证据在 Rerank 前后发生了什么变化。

这部分承担了项目的可解释性展示和答辩演示职责。

---

## 2. 主要负责的代码和文件

### Stage 1 前端

```text
frontend/
├── index.html
├── styles.css
├── app.js
└── README.md
```

主要对应 Stage 1 的：

```text
vector / graph / hybrid
```

### Stage 2 前端

```text
frontend_stage2/
├── index.html
├── styles.css
└── app.js
```

主要对应：

```text
vector / graph / hybrid / rerank / full
```

### 静态演示版本

```text
frontend_demo/
frontend_demo_clean/
```

这些目录包含静态演示数据和页面，在后端不可用或只需要展示界面时使用。静态 Demo 不能冒充实时系统测量结果。

### 集成交付副本

```text
HybridRAG_live_integration/
```

这是整理后的独立集成/演示副本。主要开发源码仍以根目录下的 `src/`、`stage2/` 和 `frontend_stage2/` 为准。

### 文档和评估协作

D 可以协助：

```text
docs/
scripts/plot_results.py
scripts/build_case_studies.py
results/figures/
```

组长负责实验设计和最终统计口径，D 负责把已确认的结果组织成清晰的界面、图表和案例材料。

---

## 3. 前端与后端的边界

Stage 2 后端由组长负责，入口是：

```text
src/backend/app_stage2.py
```

前端通过以下接口调用后端：

```text
GET  /api/health
POST /api/ask
```

请求示例：

```json
{
  "query": "Which model uses GCN as an encoder?",
  "method": "rerank",
  "verify": false
}
```

前端不直接加载 BGE-M3、FAISS 或 NetworkX，也不直接读取 `.env` 和 API Key。

---

## 4. Stage 2 返回数据

主要响应字段：

```text
query                 原始问题
method                当前检索方法
answer                最终答案
evidence              最终选中的证据
before_evidence       Rerank 前候选证据
graph                 用于可视化的节点和边
checks                引用和 Claim 检查结果
timings               Pipeline 内部耗时
request_latency_ms    完整请求耗时
```

每条 Evidence 可能包含：

```text
evidence_id
source_type
content
paper_id
paper_ids
title
section
retrieval_score
rerank_score
metadata
```

`source_type` 主要有：

```text
vector
triple
path
```

---

## 5. 前端页面负责展示什么

### 方法选择

允许用户在以下模式间切换：

```text
Vector
Graph
Hybrid
Rerank
Full
```

只有 `full` 模式需要开放额外的 Claim Verification 选项。

### 最终答案

答案区域需要保留：

- 段落和列表结构；
- `[paper_id]` 引用；
- 证据不足提示；
- 请求失败提示。

### Evidence Cards

每张证据卡片应展示：

- Vector、Triple 或 Path 类型；
- 论文标题或 `paper_id`；
- 章节；
- 证据内容摘要；
- Retrieval 或 Rerank 分数。

### Before/After Rerank

对于 `rerank` 和 `full`：

- `before_evidence` 展示候选池；
- `evidence` 展示 Rerank 后保留的证据；
- 未保留的候选可以标记为 `FILTERED BY RERANKER`。

这个视图可以直观解释 Reranker 做了什么。

### Graph Visualization

Graph 区域展示：

```text
实体节点
论文来源节点
关系边
Provenance 边
```

例如：

```text
VGAE --USES--> GCN
VGAE --PROVENANCE--> vgae_2016
```

### Verification

验证区可以展示：

- 答案引用了哪些论文；
- 哪些引用 ID 合法；
- Citation validity；
- 每条 Claim 是否获得证据支持；
- 支持该 Claim 的 Evidence ID；
- 验证请求是否失败。

---

## 6. `frontend_stage2/app.js` 的主要逻辑

```text
loadHealth()
→ 获取论文、Chunk、图节点和边数量

ask()
→ 读取问题、方法和 Verify 选项
→ POST /api/ask
→ 保存响应

renderAll()
→ 统一刷新答案、证据、图和验证信息

renderEvidence()
→ 展示 Before/After Evidence Cards

renderGraph()
→ 绘制节点和关系边

renderVerification()
→ 展示引用和 Claim 检查结果
```

文本写入 HTML 前会经过转义，避免把论文或模型输出直接当成页面 HTML 执行。

---

## 7. 与其他成员的接口

### 与组长协作

组长提供稳定的后端请求和响应格式。D 需要及时反馈：

- 缺少哪些展示字段；
- Evidence 数据是否统一；
- Graph 路径是否包含足够的节点和关系信息；
- Verification 错误是否可以和“不支持”区分显示。

### 与 A：Data 协作

论文标题、年份和来源展示依赖 A 提供的 metadata。前端不应自己维护另一套论文名称映射。

### 与 B：Vector 协作

Vector Evidence 需要展示：

```text
paper_id / title / section / content / score
```

如果某字段缺失，应通过接口修正，而不是在前端猜测。

### 与 C：Graph 协作

Graph 可视化需要：

```text
source / relation / target
path nodes / relations
paper_ids
```

前端负责展示关系，不负责重新推导图路径。

---

## 8. 文档和评估协作

D 可以协助将组长确认过的实验结果制作成：

- 方法对比表；
- 按题型分组的柱状图；
- 延迟对比图；
- Rerank 前后证据案例；
- 成功和失败案例页；
- 答辩截图和操作流程。

每个图表必须能够追溯到 `results/` 中的真实 Artifact。不要手工填写未经测量的数值。

案例展示应至少包含：

```text
问题
方法
检索证据
最终答案
引用/验证结果
为什么成功或失败
```

---

## 9. 建议讲解顺序

1. 展示页面整体布局和方法选择；
2. 输入一个示例问题；
3. 解释请求怎样发送到 `/api/ask`；
4. 展示最终答案和 `[paper_id]` 引用；
5. 展示 Vector、Triple、Path 三类 Evidence Card；
6. 切换 Before/After，说明哪些证据被 Reranker 过滤；
7. 展示图节点和关系边；
8. 展示引用和 Claim 验证；
9. 最后展示实验图表和失败案例。

### 可直接使用的简短讲解

> 我负责前端、证据展示和一部分文档评估协作。前端通过 Stage 2 后端接口提交问题和方法选择，然后展示答案、Vector/Triple/Path Evidence、Rerank 前后变化、图关系、引用检查和 Claim 验证结果。这里的重点不是只做一个聊天框，而是把系统的证据链展示出来，让用户看到答案来自哪篇论文、使用了什么关系，以及哪些候选被 Reranker 保留或过滤。我还协助把真实实验 Artifact 整理成图表和成功、失败案例。

---

## 10. 演示时适合展示什么

- `/api/health` 成功连接状态；
- 五种方法选择；
- 同一个问题在不同模式下的 Evidence 差异；
- Rerank 前后证据切换；
- `VGAE --USES--> GCN` 图关系；
- `[paper_id]` 引用及 Claim 支持信息；
- 后端失败时的明确错误提示；
- 一张成功案例和一张失败案例。

---

## 11. 容易说错的地方

- 不要把静态 Demo 数据称为实时检索结果；
- 不要在前端写死新的实验结果或论文来源；
- 不要把 Citation validity 解释为 Claim 一定正确；
- 不要隐藏验证 API 失败，失败不等于“不支持”；
- 不要把图路径动画称为模型真实思维过程；
- 不要在浏览器中读取或展示 API Key；
- 不要让界面功能描述超过后端真实能力。

---

## 12. Frontend 交付清单

- Stage 1/Stage 2 API 地址和请求字段正确；
- 方法切换逻辑正确；
- 用户输入和模型输出经过安全转义；
- Vector、Triple、Path 显示有明确区分；
- Rerank Before/After 可理解；
- 论文来源和分数显示正确；
- Verification 失败与语义不支持分开显示；
- 页面错误状态清晰；
- 图表和案例都可以追溯到真实结果文件。
