from pydantic  import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap,RunnableLambda
from typing import List
import json

class VideoAnswer(BaseModel):
    text: str = Field(description="Câu trả lời chi tiết, sử dụng định dạng Markdown (như **in đậm**, list, \n xuống dòng) để trình bày đẹp mắt.")
    filename: List[str] = Field(description="Tên file transcript gốc")
    video_url: List[str] = Field(description="URL của video gốc")
    start_timestamp: List[str] = Field(description="Thời điểm bắt đầu (format: HH:MM:SS)")
    end_timestamp: List[str] = Field(description="Thời điểm kết thúc (format: HH:MM:SS)")
    confidence: List[str] = Field(description="Độ tin cậy: zero/low/medium/high")

parser = JsonOutputParser(pydantic_object=VideoAnswer)

class Offline_RAG:
    def __init__(self, llm, retriever, reranker)-> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
        Bạn là một trợ lý AI thông minh, chuyên trả lời câu hỏi dựa trên nội dung video transcript.
        
        NHIỆM VỤ CỦA BẠN:
        Dựa vào danh sách transcript bên dưới, hãy tạo câu trả lời JSON cho câu hỏi của người dùng.
        
        1. HƯỚNG DẪN NỘI DUNG (Trường 'text'):
           - Phải dựa vào nội dung transcript để trả lời, KHÔNG ĐƯỢC BỊA CHUYỆN, tuân theo question của người dùng
           - Trả lời chi tiết và mượt mà bằng tiếng Việt.
           - **BẮT BUỘC dùng Markdown:** Sử dụng **in đậm** cho ý chính, gạch đầu dòng (-) cho danh sách.
           - **Quan trọng về xuống dòng:** Vì đầu ra là JSON, bạn phải dùng ký tự `\\n` để biểu thị xuống dòng. Tuyệt đối không dùng dấu xuống dòng thật (line break) trong chuỗi giá trị JSON.
           - **Trích dẫn:** Sau mỗi ý lấy từ video, hãy đánh số thứ tự [index] (ví dụ [0], [1]...) tương ứng với vị trí video trong mảng `video_url`.

        2. HƯỚNG DẪN DỮ LIỆU TRÍCH XUẤT (Các trường list):
           - `video_url`: Danh sách các link video đã dùng để tham khảo.
           - `start_timestamp` / `end_timestamp`: Thời điểm bắt đầu và kết thúc tương ứng của thông tin trong video đó.
           - `confidence`: Đánh giá độ tin cậy (high/medium/low).

        3. CÁC TRƯỜNG HỢP NGOẠI LỆ:
           - Nếu không biết câu trả lời: Ghi vào text là "tôi không biết hehe".
           - Nếu câu hỏi không liên quan nội dung video: Ghi vào text là "tôi chỉ được huấn luyện trả lời các câu hỏi liên quan đến nội dung video thui hihi".
           - Không bịa ra thông tin không có trong transcript.

        Định dạng đầu ra phải tuân theo JSON schema sau:
        {format_instructions}

        ----------------
        Dữ liệu Transcript đầu vào:
        {context}

        ----------------
        Câu hỏi: {question}
        """)
        self.retriever = retriever
        self.reranker = reranker



    def format_doc(self, docs,*args, **kwargs):
        formatted = []
        for doc in docs:
            url = doc.metadata.get("video_url", "")
            filename = doc.metadata.get("filename", "")
            start = doc.metadata.get("start_timestamp", "")
            end = doc.metadata.get("end_timestamp", "")
            content = json.dumps(doc.page_content)  # escape quotes, newlines
            formatted.append(f'{{"video_url": "{url}", "filename": "{filename}", "start": "{start}", "end": "{end}", "content": {content}}}')
        return "[" + ",".join(formatted) + "]"

    
    def rerank_with_query(self, docs_and_query):
        docs, query = docs_and_query
        return self.reranker.rerank(docs, query)
    
    # Hàm lấy context để đưa vào prompt 
    def get_context(self, query: str):
        docs = self.retriever.get_relevant_documents(query)
        reranked = self.reranker.rerank(docs, query)
        return self.format_doc(reranked)
    
    def get_chain(self):
        return (
            {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(self.get_context),
        }
        | self.prompt.partial(format_instructions=parser.get_format_instructions())
        | self.llm
        )
