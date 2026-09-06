import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- 配置 ---
# 如果 BAAI/bge-m3 下载太慢，就把下面这行换成 "all-MiniLM-L6-v2"
MODEL_NAME = "BAAI/bge-m3"  
DATA_PATH = "data/sample/papers.jsonl"

print("正在加载向量模型（首次运行会自动下载，请保持网络畅通）...")
model = SentenceTransformer(MODEL_NAME)

# --- 读取数据 ---
papers = []
chunk_ids = []
chunk_texts = []

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line.strip())
        papers.append(paper)
        chunk_id = f"{paper['paper_id']}_001"
        chunk_ids.append(chunk_id)
        chunk_texts.append(paper['text'])

print(f"共读取 {len(papers)} 篇论文，生成 {len(chunk_texts)} 个文本块")

# --- 生成向量并建立索引 ---
embeddings = model.encode(chunk_texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # 内积相似度
index.add(embeddings)
print(f"FAISS 索引构建完成，共 {index.ntotal} 条向量")

# --- 核心检索函数（必须严格按照这个接口） ---
def retrieve_vector(query: str, top_k: int = 5) -> list[dict]:
    query_emb = model.encode([query], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype('float32')
    
    scores, indices = index.search(query_emb, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(papers):
            results.append({
                "paper_id": papers[idx]["paper_id"],
                "chunk_id": chunk_ids[idx],
                "text": chunk_texts[idx],
                "score": float(scores[0][i])
            })
    return results

# --- 测试入口 ---
if __name__ == "__main__":
    test_query = "which model uses GCN as an encoder?"
    print(f"\n测试查询: {test_query}")
    results = retrieve_vector(test_query, top_k=3)
    
    for i, r in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  paper_id: {r['paper_id']}")
        print(f"  score: {r['score']:.4f}")
        print(f"  text 预览: {r['text'][:80]}...")