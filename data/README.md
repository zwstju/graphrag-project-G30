# 真实论文数据：两批共 12 篇

当前包含 GCN、VGAE、GraphSAGE、GAT、GIN，以及 DeepWalk、node2vec、MPNN、DiffPool、SGC、APPNP、Cluster-GCN，共 12 篇用户提供的 arXiv PDF、149 页。文本直接从真实 PDF 提取，不是示例摘要，也没有使用 LLM 改写。B、C 统一读取 `data/processed/papers.jsonl`。

## 文件说明

- `raw/`：原始 PDF，保持原文件名和内容。
- `papers_manifest.json`：论文编号、标题、作者、年份、PDF 文件名、版本链接和正文起点。元数据依据 PDF 首页人工核对；链接依据 PDF 的 arXiv 版本号生成，不代表已核实用户实际下载地址。
- `processed/papers.jsonl`：共享接口，一行一篇论文，UTF-8，无 BOM。
- `processed/paper_pages.jsonl`：辅助溯源数据，一行一页，包含 `paper_id`、从 1 开始的 PDF 页码和该页清洗文本；不是 B 的 chunk 数据。
- `processed/quality_report.json`：每篇页数、字符数、各页字符数和原始文件 SHA-256。

## 固定接口

每篇仅含五个字段：`paper_id`（唯一字符串）、`title`（完整标题）、`authors`（姓名字符串列表）、`year`（整数）、`text`（正文字符串）。正文换行以 JSON 转义形式存储，每一物理行均可独立解析。不要将整个文件作为 JSON 数组读取。

编号沿用组内示例：`gcn_2017`、`vgae_2016`、`graphsage_2017`、`gat_2018`、`gin_2019`。年份采用会议发表年份；VGAE 采用 2016 年预印本年份。GraphSAGE 文件为 2018 年修订版，但发表年份仍为 2017。编号不能随文件版本改动。

## 复现

在项目根目录执行（需要 Python 3.10 或以上）：

```sh
python -m pip install -r src/data_processing/requirements.txt
python src/data_processing/process_papers.py
```

脚本默认从自身位置定位项目，亦可使用 `--data-dir` 指定包含清单和 raw 文件夹的数据目录。重复运行会重新生成三个 processed 文件，不修改原始 PDF。先完成全部输入检查，再写出结果。

## 清洗策略

1. 使用 pypdf 读取每一页的文字层，不进行 OCR。
2. 正文从摘要开始；没有摘要的 VGAE 从第一节开始。标题、作者、单位和邮箱不作为正文前缀。
3. 删除已识别的重复会议页眉、arXiv 版本标记和位于页边界的对应页码。
4. 将常见英文连字还原，清理空字符和多余空格。保留数学符号，不全局删除数字。
5. 跨行断词仅在合并后的单词已出现在同篇论文中时去掉连字符；否则保留连字符。保守合并部分英文续行，保留其他行及页边界。
6. 保留参考文献、附录和可提取的图注，供后续检索和引用关系抽取使用。

## 质量与限制

脚本检查唯一编号、元数据类型、逐页非空文本、正文起点、基本长度和 Unicode 替换字符。字符数不是 token 数、chunk 数或质量评分。

本版用于第一阶段文字检索和实体关系抽取。PDF 图像不会变成语义描述；公式上下标、表格列关系、图中文字顺序及少量断词可能不完整。正文提取不保证版面级复原，也不适合直接自动抽取精确实验数值。VGAE 首页图注在提取顺序中位于介绍段落之前，但两者均保留。回答涉及公式、图表或数字时应对照原始 PDF。没有把程序检查等同于全文人工校对。

参考文献里的模型、数据集不一定是当前论文使用的对象，C 抽关系时必须结合正文证据，不能只按关键词共现建立关系。B、C 的集成试读仍需各自确认。

## 第二批处理与合并结果

七篇新增论文已经处理，共 84 页；原五篇为 65 页。合并后的 papers.jsonl 共 12 条，paper_pages.jsonl 共 149 条。原有五篇 JSON 记录与合并前逐条一致，使用更新脚本从全部 12 篇 PDF 重新生成也得到相同记录。原始 PDF 未改动。

新增专属修复包括 DeepWalk 模型名断裂和粘连、node2vec 的 ofnode2vec 粘连、MPNN/SGC 后续页面起始处的重复标题、SGC latexit 软件标记、DiffPool 首页预印本标记；authors 现在显式检查列表类型。无关论文不启用这些专属修复。

APPNP 使用 2022 年修订的 v6，发表年份仍为 ICLR 2019；DiffPool 使用 2019 年修订的 v4，发表年份为 NeurIPS 2018。DiffPool 年份依据：https://proceedings.neurips.cc/paper_files/paper/2018/hash/e77dbaf6759253c7c6d0efc5690369c7-Abstract.html 。作者写法与顺序依据所给 PDF 首页，SGC 作者顺序以真实 PDF 为准。

第二批已进行全页结构检查，查看每篇首页、第二页、中间页、末页的渲染缩略图并抽查相应提取文本首尾；不是全文逐句校对。部分图表文字、出版信息仍可能插入正文，跨页断词和部分空格仍可能不完整，公式与表格数值须回查原始 PDF。

统计：论文 12 篇；PDF 页面 149 页；chunk 数、实体数、关系数待 B/C 对此版本处理后提供。B/C 集成试读尚未验证。
