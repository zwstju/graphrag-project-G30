# A：Data 模块说明

## 1. 角色定位

Data 模块负责把原始科研论文转换成 Vector 和 Graph 都能稳定使用的标准语料。

一句话概括：

> Data 模块不负责回答问题，而是保证下游系统读取到的论文正文、元数据和论文 ID 是完整、稳定、可追溯的。

Data 是整个项目的入口。如果论文文本缺页、论文 ID 不稳定或 metadata 错误，后面的向量检索、知识图谱、Benchmark 和引用都会受到影响。

---

## 2. 主要负责的文件

| 文件或目录 | 作用 |
|---|---|
| `data/papers_manifest.json` | 记录论文 ID、标题、作者、年份、PDF 文件名和来源信息 |
| `data/raw/*.pdf` | 原始论文 PDF |
| `src/data_processing/process_papers.py` | PDF 文本提取、清洗、校验和标准语料生成 |
| `data/processed/papers.jsonl` | 下游模块使用的论文级标准语料 |
| `data/processed/paper_pages.jsonl` | 保留论文页级文本，便于质量追踪 |
| `data/processed/quality_report.json` | 记录页数、字符数、文件哈希和已知限制 |
| `data/README.md` | 数据目录和数据来源说明 |
| `scripts/fetch_gnn_corpus.py` | 扩展论文语料时的下载与匹配辅助脚本 |

Data 模块的正式交付物是：

```text
data/processed/papers.jsonl
```

---

## 3. 输入与输出

### 输入

```text
data/papers_manifest.json
data/raw/*.pdf
```

Manifest 中每篇论文需要有稳定信息，例如：

```json
{
  "paper_id": "vgae_2016",
  "title": "Variational Graph Auto-Encoders",
  "authors": ["Thomas Kipf", "Max Welling"],
  "year": 2016,
  "filename": "1611.07308v1.pdf",
  "source_url": "...",
  "body_start": "..."
}
```

### 输出

`papers.jsonl` 每行对应一篇论文：

```json
{
  "paper_id": "vgae_2016",
  "title": "Variational Graph Auto-Encoders",
  "authors": ["Thomas Kipf", "Max Welling"],
  "year": 2016,
  "text": "清洗后的完整论文正文"
}
```

下游约定：

- 一行只能表示一篇论文；
- `paper_id` 必须唯一；
- `text` 不能为空；
- metadata 必须与原始论文对应；
- Vector 和 Graph 必须使用同一个 `paper_id`。

---

## 4. `process_papers.py` 的处理过程

```text
读取 Manifest
   ↓
逐篇打开 PDF
   ↓
pypdf 提取每页文本
   ↓
检查空白页或扫描图片页
   ↓
清理页眉、页码和异常字符
   ↓
谨慎修复跨行断词
   ↓
从配置的正文起点截取第一页
   ↓
拼接为论文全文
   ↓
执行长度和异常字符检查
   ↓
全部论文成功后统一写出结果
```

代码没有简单地删除所有连字符，因为科研论文中很多连字符是有意义的，例如：

```text
graph-structured
Weisfeiler-Lehman
```

只有在能够从论文词汇中确认拼接结果时，才修复跨行断词。

---

## 5. 数据质量检查

Data 模块需要关注以下问题：

### PDF 是否真的可以提取文本

如果某一页提取出的文本过短，可能说明 PDF 是扫描图片或文本层异常。代码会停止处理，而不是把空内容静默写入语料。

### 是否保留了正文

第一页通常包含标题、作者和会议说明。`body_start` 用于定位正文起点，避免把论文首页的大量非正文信息混入语料。

### 是否出现乱码

代码会处理常见 ligature，例如：

```text
ﬁ → fi
ﬂ → fl
```

并检查 Unicode replacement character。

### 论文 ID 是否稳定

`paper_id` 会同时出现在：

```text
papers.jsonl
Vector chunk metadata
Graph triples
Benchmark gold_papers
最终答案引用
```

因此不能随意重命名。

### 是否保留来源信息

`quality_report.json` 中保存原始 PDF 的 SHA-256、来源 URL、页数和字符数，使语料可以追溯。

---

## 6. 与其他成员的接口

### 向 B：Vector 交付

B 从 `papers.jsonl` 读取：

```text
paper_id
title
year
text
```

之后完成检索分块和向量索引。

### 向 C：Graph 交付

C 从同一个 `papers.jsonl` 读取论文正文，将其切成图谱抽取块，再调用 LLM 抽取三元组。

### 向组长交付

组长需要确认：

- 语料实际包含多少篇论文；
- Benchmark 中的 Gold Paper ID 都存在；
- 论文来源和质量限制能够在汇报中准确说明。

### 与 D：Frontend 协作

D 展示论文标题和来源时，依赖 Data 提供的准确 metadata，但不会直接读取原始 PDF。

---

## 7. 建议讲解顺序

1. 先展示 `data/raw/` 中的原始 PDF；
2. 展示 `data/papers_manifest.json`，解释稳定 `paper_id`；
3. 打开 `process_papers.py`，说明文本提取和清洗逻辑；
4. 展示 `papers.jsonl` 中的一条记录；
5. 展示 `quality_report.json` 中的页数、字符数和哈希；
6. 解释这份语料怎样同时交给 Vector 和 Graph。

### 可直接使用的简短讲解

> 我负责 Data 模块。我的输入是原始 GNN 论文 PDF 和论文 Manifest，输出是统一的 `papers.jsonl`。处理过程中会提取每页文本，删除页眉页码，修复部分 PDF 字符和跨行断词，并检查空白页、异常长度和乱码。每篇论文都有稳定的 `paper_id`，这个 ID 会被 Vector、Graph、Benchmark 和最终引用共同使用。为了保证可追溯性，我们还保存页级文本、PDF 哈希、页数和字符数量等质量报告。

---

## 8. 演示时适合展示什么

- 一篇原始 PDF；
- Manifest 中对应的论文记录；
- `papers.jsonl` 中清洗后的同一篇论文；
- `quality_report.json` 中对应的质量信息；
- 同一个 `paper_id` 如何出现在后续 Vector chunk 和 Graph triple 中。

---

## 9. 容易说错的地方

- 不要把目标论文清单数量说成已经成功进入语料的论文数量；
- 不要声称 PDF 中的公式、表格和图片已经完整重建；
- 不要为了提高数量而静默保留空白页或解析失败论文；
- 不要随意修改 `paper_id`；
- 不要把 References 是否进入检索语料归到 PDF 清洗结果中：原始论文可保留 References，正常检索时由后续构建步骤排除。

---

## 10. Data 交付清单

- 原始 PDF 来源明确；
- Manifest metadata 完整；
- `paper_id` 唯一且稳定；
- 所有 PDF 页面经过文本可用性检查；
- `papers.jsonl` 可以被逐行解析；
- 每篇论文正文非空；
- 页级文本和质量报告已保存；
- 已知的 PDF 提取限制已记录。
