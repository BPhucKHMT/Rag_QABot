from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from generation.llm_model import get_llm
from langchain_core.prompts import PromptTemplate

# -------------------------
# RAG chain
# -------------------------
from data_loader.file_loader import Loader
from vector_store.vectorstore import VectorDB
from rag.offline_rag import Offline_RAG
from retriever.reranking import CrossEncoderReranker
from retriever.keyword_search import BM25KeywordSearch
from retriever.hybrid_search import HybridSearch

def build_rag_chain():
    llm = get_llm()
    vector_db = VectorDB()
    vector_retriever = vector_db.get_retriever()
    documents = vector_db.get_documents()
    bm25_search = BM25KeywordSearch(documents).get_retriever()
    hybrid_search = HybridSearch(vector_retriever, bm25_search).get_retriever()
    reranker = CrossEncoderReranker()
    rag_chain = Offline_RAG(llm, hybrid_search, reranker)
    return rag_chain.get_chain()

rag_chain = build_rag_chain()

# -------------------------
# Tool Prompt + Retrieve tool
# -------------------------
toolPrompt = PromptTemplate.from_template("""
Bạn là trợ lý AI cho hệ thống hỏi đáp môn học.
Với bất kỳ câu hỏi nào về machine learning, deep learning, AI sinh tạo hoặc các kiến thức liên quan, bạn ***phải sử dụng công cụ `Retrieve`*** để lấy thông tin chính xác.
Khi sử dụng công cụ `Retrieve`, hãy trích xuất một truy vấn rõ ràng, ngắn gọn từ câu hỏi của người dùng và lịch sử chat. Truy vấn này sẽ được đặt vào tham số `query` của tool.
Với các câu hỏi khác hoặc tương tác thông thường, bạn có thể trả lời trực tiếp.
\\n Người dùng là sinh viên Việt Nam, hãy trả lời hoàn toàn bằng tiếng Việt.
\\n Đây là lịch sử chat: {chat_history}
""")


class Retrieve(BaseModel):
    query: str = Field(description="should be a search query")

llm = get_llm()
agent = llm.bind_tools([Retrieve])
agent_chain = toolPrompt | agent

# -------------------------
# LangGraph State
# -------------------------
class State(BaseModel):
    chat_history: List[dict]
    agent_output: dict = {}
    response: str = ""

# -------------------------
# Node — Agent quyết định (tool hoặc direct)
# -------------------------
def node_agent(state: State):
    chat_hist_str = "\n".join([f"{m['role']} : {m['content']}" for m in state.chat_history])
    resp = agent_chain.invoke({"chat_history": chat_hist_str})

    # chuyển AIMessage sang dict
    if hasattr(resp, "content"):
        resp_dict = {"content": resp.content, "tool_calls": getattr(resp, "tool_calls", [])}
    else:
        resp_dict = resp

    return {"agent_output": resp_dict} # -> format : {"content": ..., "tool_calls": [...]}

# -------------------------
# CONDITIONAL — Xem Agent có gọi tool không
# -------------------------
def route_decision(state: State):
    out = state.agent_output
    if "tool_calls" in out and out["tool_calls"]:
        return "rag"
    return "direct"

# -------------------------
# Node 3A — Direct Answer
# -------------------------
def node_direct_answer(state: State):
    return {"response": state.agent_output["content"]}

# -------------------------
# Node 3B — RAG Answer
# -------------------------
def node_rag_answer(state: State):
    tool_call = state.agent_output["tool_calls"][0]
    query = tool_call["args"]["query"]
    rag_result = rag_chain.invoke(query)
    return {"response": rag_result}

# -------------------------
# Build Graph
# -------------------------
graph = StateGraph(State)

graph.add_node("agent", node_agent)
graph.add_node("direct", node_direct_answer)
graph.add_node("rag", node_rag_answer)

graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    route_decision,
    {
        "direct": "direct",
        "rag": "rag"
    }
)
graph.add_edge("direct", END)
graph.add_edge("rag", END)

workflow = graph.compile()

# -------------------------
# Test run
# -------------------------
if __name__ == "__main__":
    from IPython.display import display, Image
    import pprint

    chat_history = [ {"role": "user", "content": "Xin chào bạn"},
                     {"role": "assistant", "content": "Chào bạn!"},
                       {"role": "user", "content": "ý tưởng naive bayes trong tạo sinh"} ]
    out = workflow.invoke({"chat_history": chat_history})
    print(out["response"])


    state = State(chat_history=chat_history)

    # Node Agent
   # Chạy node agent
agent_out = node_agent(state)
state.agent_output = agent_out["agent_output"]

# Kiểm tra xem agent có gọi tool không
if state.agent_output.get("tool_calls"):
    tool_call = state.agent_output["tool_calls"][0]
    query = tool_call["args"]["query"]
    print("Query mà agent sẽ gửi cho RAG:", query)
else:
    print("Agent không gọi tool, trả lời trực tiếp:", state.agent_output.get("content"))