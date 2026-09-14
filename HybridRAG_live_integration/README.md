# Stage2 前后端一体化版本

这个包不会覆盖你原来的 `src/backend/app.py` 或 `frontend/`。

它新增：

- `src/backend/app_stage2.py`
- `frontend_stage2/`

后端直接调用最终 `stage2.pipeline.Stage2System`，并由 FastAPI 同时托管新前端。

## 安装到项目

在项目根目录执行：

```bash
cp src/backend/app_stage2.py ~/Projects/graphrag-project-G30/src/backend/app_stage2.py
cp -r frontend_stage2 ~/Projects/graphrag-project-G30/
```

如果你直接把整个 zip 解压到项目根目录，也可以。

## 启动

```bash
cd ~/Projects/graphrag-project-G30
source .venv/bin/activate

uvicorn src.backend.app_stage2:app \
  --host 127.0.0.1 \
  --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

只需要这一个进程，不再单独启动 `python -m http.server`。

## 方法

- `Vector` → Stage2 `vector`
- `Hybrid` → Stage2 `hybrid`
- `Hybrid + Rerank` → Stage2 `rerank`
- `Full` → Stage2 `full`

`Hybrid + Rerank` 会同时返回：
- `before_evidence`: Vanilla Hybrid evidence
- `evidence`: Rerank 后 evidence

因此新前端可以直接做 Before / After 对照。

## Claim Verification

实时 verifier 很慢且可能消耗 API。

因此默认关闭。要实时运行：
1. 选择 `Full`
2. 勾选 `Claim verification`

录屏建议使用 `Hybrid + Rerank`，展示检索、Evidence Graph、Before/After、Answer；验证部分可在 PPT 单独展示已经完成的离线 79.9% 结果。
