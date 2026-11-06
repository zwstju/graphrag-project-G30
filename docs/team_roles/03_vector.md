# B：VectorRAG 模块说明

## 1. 角色定位

Vector 模块负责把论文正文转换成可检索的文本块和向量索引，并根据用户问题返回语义最相关的论文证据。

一句话概括：

> Vector 模块接收 A 提供的标准论文语料，完成 Chunking、BGE-M3 Embedding、FAISS 索引和 `retrieve_vector`，再把文本证据交给组长的 Hybrid Pipeline。

---

## 2. 主要负责的代码和文件

### Stage 1 Vector baseline

| 文件 | 作用 |
|---|---|
| `src/vector/retriever.py` | 固定字符分块、BGE-M3 编码、内存 FAISS 索引和 `retrieve_vector()` |

Stage 1 主要用于跑通基础接口：

```python
retrieve_vector(query: str, top_k: int = 5) -> list[dict]
```

### Stage 2 Vector 实现

| 文件 | 作用 |
|---|---|
| `stage2/data/structure_chunker.py` | 根据论文结构、段落和句子进行分块 |
| `stage2/data/fixed_chunker.py` | 固定 1200 字符、200 字符重叠的对照分块 |
| `stage2/retrieval/vector.py` | BGE-M3 编码、FAISS 持久化、完整性检查和在线搜索 |
| `scripts/build_structure_chunks.py` | 生成结构感知分块 |
| `scripts/build_retrieval_chunks.py` | 排除 References，生成正常检索语料 |
| `scripts/build_fixed_chunks.py` | 生成固定分块对照语料 |
| `scripts/build_vector_index.py` | 根据配置构建并保存向量索引 |
| `scripts/smoke_test_vector_retrieval.py` | 执行少量查询的定性检索检查 |
| `configs/default.json` | 结构化 Vector 索引配置 |
| `configs/fixed_chunk.json` | 固定分块对照索引配置 |

相关测试：

```text
tests/test_fixed_chunker.py
tests/test_stage2_vector_index.py
```

---

## 3. 输入与输出

### 输入

从 A 接收：

```text
data/processed/papers.jsonl
```

主要使用字段：

```text
paper_id
title
year
text
```

### 离线输出

结构化检索使用：

```text
data/stage2/structure_chunks.jsonl
data/stage2/retrieval_chunks.jsonl
data/stage2/vector.index
data/stage2/vector_meta.jsonl
data/stage2/vector_manifest.json
```

固定分块对照组使用：

```text
data/stage2/fixed_chunks_1200_200.jsonl
data/stage2/fixed_vector_1200_200.index
data/stage2/fixed_vector_1200_200_meta.jsonl
data/stage2/fixed_vector_1200_200_manifest.json
```

### 在线输出

Stage 2 的 `VectorIndex.search()` 返回：

```python
Evidence(
    evidence_id="V1",
    source_type="vector",
    content="论文文本块",
    paper_id="vgae_2016",
    title="Variational Graph Auto-Encoders",
    section="Method",
    retrieval_score=0.82,
    metadata={"chunk_id": "vgae_2016_s0004"},
)
```

---

## 4. 分块是怎样做的

### Stage 1 固定字符分块

Stage 1 默认使用：

```text
1200 字符窗口
200 字符重叠
```

优点是简单稳定，缺点是可能在章节、段落甚至句子中间切断语义。

### Stage 2 结构感知分块

Stage 2 首先识别科研论文的标准章节：

```text
Abstract
Introduction
Related Work
Background
Method
Experiments
Results
Conclusion
Appendix
References
```

然后：

1. 在章节内部按段落组织内容；
2. 超长段落优先按句子拆分；
3. 只有无法满足长度时才按词边界回退切分；
4. 使用完整尾句形成相邻块重叠；
5. 保存 `paper_id`、`section` 和稳定 `chunk_id`。

结构化源语料包含 References，但真正建立正常检索索引前会统一排除 References。

---

## 5. BGE-M3 和 FAISS 是怎样工作的

离线构建：

```text
论文文本块
   ↓
BGE-M3 encode
   ↓
1024 维 float32 向量
   ↓
L2 normalization
   ↓
FAISS IndexFlatIP
```

在线查询：

```text
用户问题
   ↓
BGE-M3 查询向量
   ↓
L2 normalization
   ↓
FAISS inner-product search
   ↓
Top-k 文本块
```

文档向量和查询向量都经过 L2 归一化，所以 `IndexFlatIP` 的内积排序等价于余弦相似度排序。

BGE-M3 是嵌入模型，不是负责生成答案的聊天大模型。

---

## 6. 为什么需要持久化索引和 Manifest

Stage 1 每次新进程可能重新编码全部论文，速度慢，也难以保证实验使用同一个索引。

Stage 2 将索引保存到磁盘，并使用 Manifest 记录：

- 嵌入模型名称；
- 构建设备和 batch size；
- 语料 SHA-256；
- chunk 数量；
- 向量维度；
- 是否归一化；
- FAISS 索引类型；
- 索引和 metadata 文件哈希。

加载时会检查：

```text
index.ntotal == metadata_count
模型一致
向量维度一致
语料哈希一致
索引和 metadata 哈希一致
```

这样可以避免“代码读取了新语料，却仍使用旧索引”等静默错误。

---

## 7. 与其他成员的接口

### 从 A：Data 接收

```text
data/processed/papers.jsonl
```

Vector 模块不修改论文 ID，只为每个文本块生成稳定 `chunk_id`。

### 向组长交付

Stage 1 返回字典列表，Stage 2 返回统一 `Evidence` 列表。

组长只需要调用检索接口，不需要重新实现 BGE-M3 或 FAISS。

### 与 C：Graph 的关系

Vector 和 Graph 是平行的候选证据生成器：

```text
Vector → 论文文本块
Graph  → 三元组和路径
```

两者不能使用不同的论文语料范围，否则 Hybrid 对比不公平。

### 与 D：Frontend 的关系

前端展示 Vector Evidence 时会使用：

```text
title
paper_id
section
content
retrieval_score
rerank_score
```

---

## 8. 建议讲解顺序

1. 解释为什么论文不能整篇直接做检索；
2. 对比固定字符分块和结构感知分块；
3. 展示一条包含 `section` 和 `chunk_id` 的结构化 chunk；
4. 解释 BGE-M3 如何把文本和问题映射到同一向量空间；
5. 解释归一化向量加 `IndexFlatIP` 等价于余弦相似度；
6. 介绍索引持久化和 Manifest 校验；
7. 展示 `VectorIndex.search()` 如何返回统一 Evidence；
8. 展示一个成功案例和一个 Vector 失败案例。

### 可直接使用的简短讲解

> 我负责 VectorRAG。A 提供清洗后的 `papers.jsonl`，我先把论文切成可检索文本块。Stage 1 使用固定字符分块，Stage 2 增加了章节、段落和句子边界感知。随后使用 BGE-M3 生成归一化的 1024 维向量，并保存到 FAISS `IndexFlatIP`。查询时只需要编码用户问题并搜索 Top-k，不会重新编码全部语料。每条结果都保留论文 ID、章节、chunk ID 和分数，再以统一 Evidence 形式交给 Hybrid Pipeline。

---

## 9. 演示时适合展示什么

- 同一篇论文的固定分块和结构化分块对比；
- `retrieval_chunks.jsonl` 中一条 Method 或 Experiments 证据；
- `vector_manifest.json` 中的模型、维度、chunk 数和哈希；
- “Which model uses GCN as an encoder?” 的 VGAE 检索结果；
- GCN 与 GAT 比较问题中 Vector 返回 APPNP 噪声的失败案例。

---

## 10. 容易说错的地方

- 不要把 BGE-M3 称为答案生成模型；
- 不要把内积和余弦相似度直接画等号，只有归一化后才等价；
- 不要把结构化源分块数和排除 References 后的检索分块数混淆；
- 不要只看相似度高就认为证据一定正确；
- 不要把 Paper-level retrieval proxy 当作段落级相关性标注；
- 不要声称 Vector 对所有关系和比较问题都优于 Graph。

---

## 11. Vector 交付清单

- Chunk 具有稳定 `paper_id` 和 `chunk_id`；
- 正常检索索引不含 References；
- 文档和查询向量使用同一模型及归一化方式；
- 索引、metadata 和 Manifest 数量一致；
- 显式 CUDA 请求失败时不会静默改用 CPU；
- 在线检索加载已有索引，不重新编码语料；
- 返回结果满足组长约定的接口。
