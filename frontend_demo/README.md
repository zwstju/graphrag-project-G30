# Evidence-Centric HybridRAG 答辩录屏版前端

这是一个完全本地、无 API 依赖的“真实实验结果回放”页面。

数据来源：
- generation_outputs.jsonl 中 C01 的三种方法真实结果
- citation_outputs.jsonl 中 C01 的真实 post-hoc claim verification
- 项目已测全局指标（Vector 0.8734 / Hybrid 0.7969 / Rerank 0.8250 / Claim support 79.9%）

## 启动

```bash
cd frontend_demo
python -m http.server 5173
```

浏览器打开：

http://127.0.0.1:5173

点击“播放演示”。

## 录屏建议

- 浏览器缩放 90%～100%
- 窗口尽量 16:9
- 全屏录 15～20 秒即可
- 页面按顺序展示：
  1. Hybrid retrieval
  2. Evidence reranking
  3. Grounded generation
  4. Claim verification

说明：该页面是 canonical Core32 artifact 的可视化回放，不会再次调用 DeepSeek，也不会伪造实时 verifier。
