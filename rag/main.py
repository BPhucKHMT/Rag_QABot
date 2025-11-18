from pydantic import BaseModel, Field

from data_loader.file_loader import Loader
from vector_store.vectorstore import VectorDB
from rag.offline_rag import Offline_RAG
from retriever.reranking import CrossEncoderReranker
from retriever.keyword_search import BM25KeywordSearch
from retriever.hybrid_search import HybridSearch
from generation.llm_model import get_llm


def build_rag_chain():
    # llm
    llm = get_llm()
    # vector db
    vector_db = VectorDB()
    #search retriever
    vector_retriever = vector_db.get_retriever()
    documents = vector_db.get_documents()
    bm25_search = BM25KeywordSearch(documents).get_retriever()
    hybrid_search = HybridSearch(vector_retriever, bm25_search).get_retriever()

    # reranker
    reranker = CrossEncoderReranker()
    #chain 
    rag_chain = Offline_RAG(llm, hybrid_search, reranker)
    return rag_chain.get_chain()

if __name__ == "__main__":
    rag_chain = build_rag_chain()
    response = rag_chain.invoke("diffusion bị gì để có lantent diffusion")
    print(response)
