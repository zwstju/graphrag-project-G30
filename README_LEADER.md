# Leader Module Quick Start

这组文件对应组长负责的部分：

```text
Query Processing
HybridRAG
Prompt Builder
LLM Client
Backend API
Integration Adapters
Architecture / Paper Notes
```

## 1. Copy

将本包中的文件复制到项目根目录。

注意：

- 不要覆盖组员已经实现的 Vector / Graph 代码。
- `src/retrieval/adapters.py` 只是负责连接 B/C 的实现。

---

## 2. Install

```bash
pip install -r requirements-leader.txt
```

---

## 3. Environment

```bash
cp .env.example .env
```

开发阶段默认：

```text
ALLOW_MOCK_RETRIEVERS=true
LLM_MOCK_MODE=true
```

因此即使 B/C 和真实 LLM 尚未接入，也可以先跑通 Backend。

---

## 4. Start Backend

项目根目录执行：

```bash
uvicorn src.backend.app:app --reload --host 0.0.0.0 --port 8000
```

检查：

```bash
curl http://127.0.0.1:8000/health
```

应返回：

```json
{"status":"ok"}
```

测试 Hybrid：

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "VGAE 和 GCN 有什么关系？",
    "mode": "hybrid"
  }'
```

---

## 5. Connect B's Vector Retriever

默认期待 B 提供：

```python
# src/vector/retriever.py

def retrieve_vector(
    query: str,
    top_k: int = 5
) -> list[dict]:
    ...
```

如果 B 的模块位置不同，例如：

```text
src/rag/vector_retriever.py
```

修改 `.env`：

```text
VECTOR_RETRIEVER_MODULE=src.rag.vector_retriever
VECTOR_RETRIEVER_FUNCTION=retrieve_vector
```

接通以后：

```text
ALLOW_MOCK_RETRIEVERS=false
```

---

## 6. Connect C's Graph Retriever

默认期待 C 提供：

```python
# src/graph/retriever.py

def retrieve_graph(
    query: str,
    top_k: int = 5
) -> dict:
    ...
```

如果位置不同，在 `.env` 中调整：

```text
GRAPH_RETRIEVER_MODULE=...
GRAPH_RETRIEVER_FUNCTION=retrieve_graph
```

---

## 7. Connect Real LLM

设置：

```text
LLM_MOCK_MODE=false
LLM_API_BASE=https://your-provider.example/v1
LLM_API_KEY=your-key
LLM_MODEL=your-model
```

本实现调用 OpenAI-compatible：

```text
POST {LLM_API_BASE}/chat/completions
```

不要把 `.env` commit 到 Git。

---

## 8. Test

```bash
pytest -q
```

---

## 9. Backend Contract

Frontend 请求：

```json
{
  "query": "VGAE 和 GCN 有什么关系？",
  "mode": "hybrid"
}
```

返回：

```json
{
  "query": "...",
  "mode": "hybrid",
  "answer": "...",
  "vector_results": [],
  "graph_results": {
    "entities": [],
    "triples": [],
    "paths": [],
    "paper_ids": []
  },
  "references": []
}
```

---

## 10. Git

你当前应该位于：

```text
feature/hybrid-rag
```

确认：

```bash
git branch --show-current
```

添加：

```bash
git add .
git status
git commit -m "feat: implement hybrid pipeline and backend skeleton"
git push
```

不要直接在 `main` 上开发。
