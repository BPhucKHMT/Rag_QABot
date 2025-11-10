# scripts/reranking.py
# pip install -U langchain langchain-chroma langchain-community langchain-huggingface transformers accelerate chromadb

import torch
from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3", model_kwargs={"device": "cpu"}
)

from langchain_chroma import Chroma

vector_db = Chroma(embedding_function=embedding, persist_directory="./database")

retriever = vector_db.as_retriever(
    search_type="mmr", search_kwargs={"k": 40, "fetch_k": 80, "lambda_mult": 0.3}
)


def retrieve_docs(ret, query: str):
    """Tương thích LangChain 0.2+ (invoke) và bản cũ (get_relevant_documents)."""
    try:
        return ret.invoke(query)  # LangChain >= 0.2
    except AttributeError:
        try:
            return ret.get_relevant_documents(query)  # bản cũ
        except AttributeError:
            return ret._get_relevant_documents(query)  # fallback


query = "Tại sao việc tự huấn luyện mô hình CLIP từ đầu là khó khả thi?"
candidates = retrieve_docs(retriever, query)


rerank_model_name = "BAAI/bge-reranker-base"
tok = AutoTokenizer.from_pretrained(rerank_model_name)
reranker = AutoModelForSequenceClassification.from_pretrained(rerank_model_name)
reranker.eval()


@torch.no_grad()
def batch_crossencoder_scores(
    q: str, texts: List[str], batch_size: int = 16, max_len: int = 512
) -> List[float]:
    scores = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tok(
            [q] * len(batch),
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )
        logits = reranker(**inputs).logits.squeeze(-1)  # [B]
        scores.extend(logits.tolist())
    return scores


texts = [d.page_content for d in candidates]
scores = batch_crossencoder_scores(query, texts, batch_size=16, max_len=512)

ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)

BAD_HINTS = ("Cảm ơn các bạn đã xem", "đăng ký kênh", "subscribe", "like và share")
final_docs = []
for d, s in ranked:
    if all(h.lower() not in d.page_content.lower() for h in BAD_HINTS):
        final_docs.append((d, s))
    if len(final_docs) >= 10:
        break

# 7) In kết quả
for i, (d, s) in enumerate(final_docs, 1):
    m = d.metadata
    print(f"[{i}] score={s:.3f} | {m.get('title', '')} | {m.get('video_url', '')}")
    print(f"    {m.get('start_timestamp', '?')} → {m.get('end_timestamp', '?')}")
    text = d.page_content.replace("\n", " ").strip()
    print("   " + (text[:300] + ("..." if len(text) > 300 else "")))
    print("-" * 80)
