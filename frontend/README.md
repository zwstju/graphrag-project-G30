# Frontend

## 周六～周一已完成
- 周六：输入框、模式选择、回答区、证据区、示例问题。
- 周日：接通 `GET /health` 和 `POST /ask`，加入加载状态与错误提示。
- 周一：展示 Vector evidence、Graph entities/triples/paths、References，并支持 Vector/Graph/Hybrid 三种模式。

## 运行
先启动后端：
```bash
cd ~/Projects/graphrag-project-G30
source .venv/bin/activate
uvicorn src.backend.app:app --reload --port 8000
```

再启动前端：
```bash
cd ~/Projects/graphrag-project-G30/frontend
python -m http.server 5173
```

浏览器访问：
`http://127.0.0.1:5173`

前端默认请求：
`http://127.0.0.1:8000`

调用链：
`浏览器 -> fetch POST /ask -> FastAPI -> HybridRAG -> JSON -> 页面展示`

当前后端需要允许本地跨域访问；你已有的 `allow_origins=["*"]` 可以用于开发阶段。
