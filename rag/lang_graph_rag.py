from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field
from typing import List
from generation.llm_model import get_llm
from langchain_core.prompts import PromptTemplate
import json  # <<< THÊM
import re    # <<< THÊM
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
# -------------------------
# -------------------------
# -------------------------
# Helper Function: Convert timestamp (M:SS or H:M:SS) to total seconds
# -------------------------
def timestamp_to_seconds(timestamp: str) -> int:
    """Chuyển đổi chuỗi timestamp (ví dụ: '0:01:28') thành tổng số giây."""
    parts = list(map(int, timestamp.split(':')))
    
    seconds = 0
    if len(parts) == 3: # H:M:S
        seconds += parts[0] * 3600
        seconds += parts[1] * 60
        seconds += parts[2]
    elif len(parts) == 2: # M:S
        seconds += parts[0] * 60
        seconds += parts[1]
    return seconds

# -------------------------
# Node 3B — RAG Answer (FIX CUỐI CÙNG CHO VẤN ĐỀ GẮN LINK/INDEX)
# -------------------------
def node_rag_answer(state: State):
    tool_call = state.agent_output["tool_calls"][0]
    query = tool_call["args"]["query"]
    
    rag_result = rag_chain.invoke(query)
    
    if hasattr(rag_result, 'content'):
        raw_content = rag_result.content
    else:
        raw_content = str(rag_result)

    # Parse JSON mạnh mẽ
    json_string = raw_content.strip()
    json_string = re.sub(r'^\s*```json\s*|\s*```\s*$|\s*undefined\s*$', '', json_string, flags=re.MULTILINE).strip()
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return {"response": f"Xin lỗi, có lỗi xảy ra khi xử lý nguồn thông tin (JSONDecodeError: {e}).\nNội dung thô: {raw_content}"}

    return {"response": data}  # Trả về dict có key 'text', 'video_url', ...

'''
def node_rag_answer(state: State):
    tool_call = state.agent_output["tool_calls"][0]
    query = tool_call["args"]["query"]
    
    rag_result = rag_chain.invoke(query)
    
    if hasattr(rag_result, 'content'):
        raw_content = rag_result.content
    else:
        raw_content = str(rag_result)
    return {"response": raw_content}


    # 1. LÀM SẠCH CHUỖI JSON MẠNH MẼ VÀ PARSE
    json_string = raw_content.strip()
    json_string = re.sub(r'^\s*```json\s*|\s*```\s*$|\s*undefined\s*$', '', json_string, flags=re.MULTILINE).strip()
    
    try:
        data = json.loads(json_string)
    except json.JSONDecodeError as e:
        return {"response": f"Xin lỗi, có lỗi xảy ra khi xử lý nguồn thông tin (JSONDecodeError: {e}). Nội dung thô: {raw_content}"}
        
    raw_text = data['text']
    
    # Lấy danh sách các nguồn từ dictionary data
    video_urls = data.get('video_url', [])
    start_timestamps = data.get('start_timestamp', [])
    end_timestamps = data.get('end_timestamp', [])

    # 2. TÁCH VĂN BẢN VÀ GẮN CITATION
    
    # RegEx để tách chuỗi. Bắt toàn bộ phần ([Video URL]: ... [End]: ...).
    # Không cần bắt các nhóm con vì ta lấy dữ liệu từ data['...'].
    # (?:...) tạo thành một non-capturing group, nghĩa là parts[] sẽ chỉ chứa các đoạn văn bản.
    citation_delimiter_pattern = r'\s*\([,.]?\s*(?:\[Video URL\]:\s*.*?\s*\[Start\]:\s*.*?\s*\[End\]:\s*.*?)\)\s*[.,]?\s*'
    
    # parts = [Text1, Text2, Text3, ...] (Không chứa các trích dẫn nữa)
    text_segments = re.split(citation_delimiter_pattern, raw_text)

    formatted_output = []
    citation_index = 0
    
    # 3. GẮN LINK (Sử dụng citation_index)
    
    # Duyệt qua các đoạn văn bản đã được tách (Text1, Text2, ...)
    for segment in text_segments:
        clean_segment = segment.strip()
        
        # Nếu segment này rỗng, đó là do có nhiều trích dẫn liền kề nhau
        if not clean_segment:
            continue

        # Thêm đoạn văn bản vào kết quả
        current_block = clean_segment
        
        # Kiểm tra xem có trích dẫn tương ứng để gắn vào block này không
        # Giả định: Số lượng segments có ý nghĩa (không rỗng) bằng số lượng citations.
        if citation_index < len(video_urls):
            
            video_url = video_urls[citation_index]
            start_time_str = start_timestamps[citation_index]
            end_time_str = end_timestamps[citation_index]
            
            start_sec = timestamp_to_seconds(start_time_str)
            source_link = f"{video_url}&t={start_sec}s"
            source_citation = f"\nNguồn: {source_link} (Từ {start_time_str} đến {end_time_str})"
            
            # Gắn link vào đoạn văn bản
            current_block += source_citation
            
            citation_index += 1
        
        formatted_output.append(current_block)

    # Lọc và tạo chuỗi kết quả cuối cùng
    final_response_string = "\n\n".join([s for s in formatted_output if s.strip()])

    return {"response": final_response_string}
    '''
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
# Hàm chính để export (Được app.py import)
# -------------------------
def call_agent(chat_history: List[dict]) -> str:
    """
    Chạy LangGraph workflow với lịch sử chat.
    """
    # Khởi tạo trạng thái ban đầu
    initial_state = State(chat_history=chat_history)
    
    # Chạy workflow và nhận trạng thái cuối cùng
    final_state = workflow.invoke(initial_state)
    
    return final_state["response"]

# -------------------------
# Test run (chỉ chạy khi chạy file trực tiếp)
# -------------------------
if __name__ == "__main__":
    from IPython.display import display, Image
    import pprint

    # Dữ liệu test 1: Hỏi về chủ đề RAG (Nên gọi tool)
    chat_history_rag = [ 
                        {"role": "user", "content": "diffusion bị gì để có lantent diffusion"}
                       ]
    print("--- TEST 1: RAG Question ---")
    out_rag = workflow.invoke({"chat_history": chat_history_rag})
    print("Final Response:", out_rag["response"])
    print("---" * 10)

    # Dữ liệu test 2: Hỏi về chủ đề thông thường (Nên trả lời trực tiếp)
    chat_history_direct = [
                        {"role": "user", "content": "cnn đã đạt thành tựu gì"} 
                       ]
    print("--- TEST 2: Direct Question ---")
    out_direct = workflow.invoke({"chat_history": chat_history_direct})
    print("Final Response:", out_direct["response"])
    print("---" * 10)
    
    
    # --- Code kiểm thử thủ công bị lỗi NameError ---
    # *Đây là các dòng bạn cần XÓA hoặc BỎ QUA trong file gốc của bạn*
    # state = State(chat_history=chat_history)
    # # Node Agent
    # # Chạy node agent
    # agent_out = node_agent(state)
    # state.agent_output = agent_out["agent_output"]
    # # Kiểm tra xem agent có gọi tool không
    # if state.agent_output.get("tool_calls"):
    #     tool_call = state.agent_output["tool_calls"][0]
    #     query = tool_call["args"]["query"]
    #     print("Query mà agent sẽ gửi cho RAG:", query)
    # else:
    #     print("Agent không gọi tool, trả lời trực tiếp:", state.agent_output.get("content"))