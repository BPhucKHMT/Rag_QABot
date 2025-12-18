from pydantic  import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap,RunnableLambda
from typing import List
import json

class VideoAnswer(BaseModel):
    text: str = Field(description="Câu trả lời cho đáp án của người dùng, đúng trọng tâm, nhưng dựa vào transcript (không bịa ra cái không có) sử dụng định dạng Markdown (như **in đậm**, list, xuống dòng) để trình bày đẹp mắt.")
    filename: List[str] = Field(description="Tên file transcript gốc")
    video_url: List[str] = Field(description="URL của video gốc, số video phải khớp với số timestamp")
    title: List[str] = Field(description="Tiêu đề của video gốc, số lượng phải khớp với số lượng timestamp")
    start_timestamp: List[str] = Field(description="Thời điểm bắt đầu (format: HH:MM:SS)")
    end_timestamp: List[str] = Field(description="Thời điểm kết thúc (format: HH:MM:SS)")
    confidence: List[str] = Field(description="Độ tin cậy: zero/low/medium/high")

parser = JsonOutputParser(pydantic_object=VideoAnswer)

class Offline_RAG:
    def __init__(self, llm, retriever, reranker)-> None:
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên video transcript. Trả lời ngắn gọn, súc tích, đúng trọng tâm.

═══════════════════════════════════════════════════════════════════
🎯 QUY TẮC VÀNG - KHÔNG ĐƯỢC VI PHẠM
═══════════════════════════════════════════════════════════════════

1. CHỈ dùng thông tin CÓ TRONG transcript bên dưới
2. KHÔNG suy luận, bổ sung, hoặc dùng kiến thức ngoài
3. Không biết → trả tuân thủ format json với trường text là  "tôi không biết hehe"
4. Không liên quan → trả tuân thủ format json với trường text là "tôi chỉ được huấn luyện trả lời các câu hỏi liên quan đến nội dung video thui hihi"

═══════════════════════════════════════════════════════════════════
📝 QUY TẮC JSON - QUAN TRỌNG ĐỂ TRÁNH LỖI
═══════════════════════════════════════════════════════════════════

⚠️ CRITICAL - JSON ESCAPING:
1. Xuống dòng: Dùng \\n (gõ: backslash + n)
   ✅ ĐÚNG: "Line 1\\nLine 2"
   ❌ SAI: "Line 1\nLine 2"

2. Backslash trong text: Dùng \\\\
   ✅ ĐÚNG: "Path C:\\\\folder"
   ❌ SAI: "Path C:\\folder"

3. Dấu ngoặc kép: Dùng \\"
   ✅ ĐÚNG: "He said \\"hello\\""
   ❌ SAI: "He said "hello""

4. Markdown an toàn:
   ✅ ĐƯỢC DÙNG: **in đậm**, *in nghiêng*, - danh sách, ### tiêu đề
   ⚠️ TRÁNH: Code blocks (```), dấu ` backtick

5. LaTeX: Escape backslash
   ✅ ĐÚNG: "$$E = mc^2$$" hoặc "$\\\\frac{{a}}{{b}}$"
   ❌ SAI: "$\\frac{{a}}{{b}}$"

═══════════════════════════════════════════════════════════════════
📋 QUY TRÌNH 3 BƯỚC - REMAP CITATION
═══════════════════════════════════════════════════════════════════

🔹 BƯỚC 1: VIẾT TEXT VỚI CITATION GỐC
Đọc transcript và viết câu trả lời với số citation theo index ban đầu.

Ví dụ: Có transcript [0], [1], [2], [3], [4]
Text: "ResNet50 có 50 layers [0]. Skip connection giúp giải quyết vanishing gradient [2]. Batch normalization [4] tăng tốc training."

🔹 BƯỚC 2: TẠO DANH SÁCH VIDEO_URL (CHỈ VIDEO ĐÃ DÙNG)
Liệt kê các video ĐÃ ĐƯỢC TRÍCH DẪN trong text, LOẠI BỎ video không dùng.

Từ ví dụ trên: Đã dùng [0], [2], [4]
→ video_url = ["url_của_transcript_0", "url_của_transcript_2", "url_của_transcript_4"]

🔹 BƯỚC 3: REMAP LẠI CITATION TRONG TEXT
Thay ĐỔI số citation để khớp với index MỚI trong video_url.

Mapping:
- [0] trong text → [0] (vì url_của_transcript_0 là phần tử đầu trong video_url)
- [2] trong text → [1] (vì url_của_transcript_2 là phần tử thứ 2 trong video_url)
- [4] trong text → [2] (vì url_của_transcript_4 là phần tử thứ 3 trong video_url)

Text cuối: "ResNet50 có 50 layers [0]. Skip connection giúp giải quyết vanishing gradient [1]. Batch normalization [2] tăng tốc training."

═══════════════════════════════════════════════════════════════════
📌 VÍ DỤ ĐẦY ĐỦ
═══════════════════════════════════════════════════════════════════

INPUT: 5 transcript [0], [1], [2], [3], [4]

❌ SAI - Không REMAP:
{{
  "text": "ResNet50 [0] và VGG [2] là hai kiến trúc phổ biến.",
  "video_url": ["url0", "url1", "url2", "url3", "url4"]
}}
Lỗi: video_url chứa cả video không dùng + citation không khớp

✅ ĐÚNG - Có REMAP:
{{
  "text": "ResNet50 [0] và VGG [1] là hai kiến trúc phổ biến.",
  "video_url": ["url_của_transcript_0", "url_của_transcript_2"],
  "filename": ["file0.txt", "file2.txt"],
  "title": ["Video về ResNet", "Video về VGG"],
  "start_timestamp": ["00:01:23", "00:02:45"],
  "end_timestamp": ["00:03:45", "00:04:12"],
  "confidence": ["high", "high"]
}}

═══════════════════════════════════════════════════════════════════
💡 HƯỚNG DẪN VIẾT TEXT
═══════════════════════════════════════════════════════════════════

1. Ngôn ngữ: Tiếng Việt tự nhiên, mạch lạc
2. Format:
   - Dùng **in đậm** cho điểm chính
   - Dùng - cho danh sách
   - Dùng \\n để xuống dòng (VÍ DỤ: "Dòng 1\\nDòng 2")
3. Citation: Đánh [0], [1], [2]... sau khi REMAP
4. Độ dài: Trả lời đủ nhưng súc tích (3-8 câu)

VÍ DỤ TEXT FORMATTING:
"**ResNet50** là kiến trúc deep learning nổi tiếng [0].\\n\\nĐặc điểm chính:\\n- Có 50 layers\\n- Sử dụng skip connection [1]\\n- Training nhanh hơn VGG"

═══════════════════════════════════════════════════════════════════
📊 CẤU TRÚC OUTPUT
═══════════════════════════════════════════════════════════════════

Tất cả các mảng PHẢI CÙNG ĐỘ DÀI và TƯƠNG ỨNG theo index:
- video_url[0] ↔ filename[0] ↔ title[0] ↔ start_timestamp[0] ↔ end_timestamp[0] ↔ confidence[0]
- video_url[1] ↔ filename[1] ↔ title[1] ↔ start_timestamp[1] ↔ end_timestamp[1] ↔ confidence[1]

Confidence levels:
- "high": Thông tin rõ ràng, trực tiếp
- "medium": Thông tin gián tiếp, cần suy luận nhẹ
- "low": Thông tin mơ hồ hoặc không đầy đủ
- "zero": Không tìm thấy thông tin

═══════════════════════════════════════════════════════════════════
📥 DỮ LIỆU ĐẦU VÀO
═══════════════════════════════════════════════════════════════════

Transcript Array (JSON format):
{context}

Câu hỏi của người dùng:
{question}

═══════════════════════════════════════════════════════════════════
🎯 YÊU CẦU ĐẦU RA
═══════════════════════════════════════════════════════════════════

{format_instructions}

⚠️ NHẮC NHỞ QUAN TRỌNG CUỐI:
1. Đã REMAP lại tất cả số [X] trong text chưa?
2. video_url chỉ chứa video ĐÃ DÙNG chưa?
3. Đã dùng \\n thay vì enter thật chưa?
4. Tất cả mảng cùng độ dài chưa?
5. JSON escape đúng chưa? (\\n, \\\\, \\")

Bắt đầu trả lời bằng JSON hợp lệ ngay:
""")
        self.retriever = retriever
        self.reranker = reranker



    def format_doc(self, docs,*args, **kwargs):
        formatted = []
        for doc in docs:
            url = doc.metadata.get("video_url", "")
            filename = doc.metadata.get("filename", "")
            title = doc.metadata.get("title", "")
            start = doc.metadata.get("start_timestamp", "")
            end = doc.metadata.get("end_timestamp", "")
            content = json.dumps(doc.page_content)  # escape quotes, newlines
            formatted.append(f'{{"video_url": "{url}", "filename": "{filename}", "title": "{title}","start": "{start}", "end": "{end}",  "content": {content}}}')
        return "[" + ",".join(formatted) + "]"


    
    # Hàm lấy context để đưa vào prompt 
    def get_context(self, query: str):
        import time
        start_time = time.time()
        docs = self.retriever.get_relevant_documents(query)
        reranked = self.reranker.rerank(docs, query)
        end_time = time.time()
        print(f"Time taken to get context: {end_time - start_time} seconds")
        return self.format_doc(reranked)
    
    def get_chain(self):
        return (
            {
            "question": RunnablePassthrough(),
            "context": RunnableLambda(self.get_context),
        }
        | self.prompt.partial(format_instructions=parser.   get_format_instructions())
        | self.llm
        )
