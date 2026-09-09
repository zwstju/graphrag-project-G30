import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# --- 配置 ---
# 如果 BAAI/bge-m3 下载太慢，就把下面这行换成 "all-MiniLM-L6-v2"
MODEL_NAME = "BAAI/bge-m3"
DATA_PATH = "data/processed/papers.jsonl"

# --- Chunking 配置 ---
CHUNK_SIZE = 1200   # 每个 chunk 的字符数
OVERLAP = 200       # 相邻 chunk 之间的重叠字符数

print("正在加载向量模型（首次运行会自动下载，请保持网络畅通）...")
model = SentenceTransformer(MODEL_NAME)


# --- 文本分块函数 ---
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """
    将长文本按字符切分成多个重叠的 chunk
    """
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += (chunk_size - overlap)
        # 如果最后一次 start 超出了文本末尾，并且已经添加了最后一个 chunk，则退出
        if start >= len(text):
            break
    
    # 确保最后一个 chunk 不重复
    if chunks and len(chunks[-1]) < 100 and len(chunks) > 1:
        # 如果最后一个 chunk 太短，可以适当往前扩展（这里简单处理，直接保留）
        pass
    
    return chunks


# --- 读取数据并生成分块 ---
chunk_meta = []     # 存储每个 chunk 的元数据: paper_id, chunk_id, title, text
chunk_texts = []    # 存储每个 chunk 的文本（用于 embedding）

print(f"正在读取数据: {DATA_PATH}")
with open(DATA_PATH, 'r', encoding='utf-8') as f:
    for line in f:
        paper = json.loads(line.strip())
        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("title", "")
        text = paper.get("text", "")
        
        # 如果 text 为空，跳过
        if not text:
            continue
        
        # 切分文本
        chunks = chunk_text(text)
        
        # 为每个 chunk 生成元数据
        for i, chunk_txt in enumerate(chunks, 1):
            chunk_id = f"{paper_id}_{i:03d}"
            chunk_meta.append({
                "paper_id": paper_id,
                "chunk_id": chunk_id,
                "title": title,
                "text": chunk_txt
            })
            chunk_texts.append(chunk_txt)

print(f"共读取 {len(set(m['paper_id'] for m in chunk_meta))} 篇论文，切分为 {len(chunk_texts)} 个文本块")


# --- 生成向量并建立索引 ---
print("正在生成向量（首次运行会慢一些，请耐心等待）...")
embeddings = model.encode(chunk_texts, normalize_embeddings=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # 内积相似度（配合归一化向量 = 余弦相似度）
index.add(embeddings)
print(f"FAISS 索引构建完成，共 {index.ntotal} 条向量")


# --- 核心检索函数（严格按照组长规定的接口） ---
def retrieve_vector(query: str, top_k: int = 5) -> list[dict]:
    """
    根据查询文本，返回最相关的 top_k 个 chunk

    参数:
        query: 用户问题
        top_k: 返回结果数量

    返回:
        list[dict]: 每个元素包含 paper_id, chunk_id, title, text, score
    """
    # 将查询也转为向量
    query_emb = model.encode([query], normalize_embeddings=True)
    query_emb = np.array(query_emb).astype('float32')

    # 在 FAISS 中搜索
    scores, indices = index.search(query_emb, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1 and idx < len(chunk_meta):
            meta = chunk_meta[idx]
            results.append({
                "paper_id": meta["paper_id"],
                "chunk_id": meta["chunk_id"],
                "title": meta["title"],
                "text": meta["text"],
                "score": float(scores[0][i])
            })

    return results


# --- 测试入口 ---
if __name__ == "__main__":
    test_query = "which model uses GCN as an encoder?"
    print(f"\n测试查询: {test_query}")
    print("-" * 50)
    
    results = retrieve_vector(test_query, top_k=5)
    
    for i, r in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  paper_id: {r['paper_id']}")
        print(f"  chunk_id: {r['chunk_id']}")
        print(f"  title: {r['title']}")
        print(f"  score: {r['score']:.4f}")
        print(f"  text 预览: {r['text'][:120]}...")