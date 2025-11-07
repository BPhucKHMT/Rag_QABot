from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain import hub

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import re

import torch
import time 


from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

model_name = "BAAI/bge-m3"            # đa ngôn ngữ, gọn nhẹ, khuyên dùng

embedding = HuggingFaceEmbeddings(model_name=model_name,model_kwargs={"device": "cuda"})

## lưu vào db
vector_db = Chroma( embedding_function= embedding, persist_directory="./chroma_bge_db")
retriever = vector_db.as_retriever(search_type="similarity", search_kwargs={"k": 3})


nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

MODEL_NAME = "Qwen/Qwen3-0.6B"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.float16,   # dùng float16 cho GPU
    device_map="auto"      # tự động đặt model lên GPU
)
import torch
print(torch.cuda.is_available())          # True nghĩa là có GPU
print(next(model.parameters()).device)    # in ra cuda:0 nếu model đang ở GPU
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True, trust_remote_code=True)

model_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    pad_token_id=tokenizer.eos_token_id,
)

llm = HuggingFacePipeline(pipeline=model_pipeline)


from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableMap,RunnableLambda


class VideoAnswer(BaseModel):
    text: str = Field(description="Câu trả lời tóm tắt trong 3 câu")
    video_url: str = Field(description="URL của video gốc")
    start_timestamp: str = Field(description="Thời điểm bắt đầu (format: HH:MM:SS)")

parser = JsonOutputParser(pydantic_object=VideoAnswer)

# ===== Prompt =====
prompt = ChatPromptTemplate.from_template("""
Dựa vào transcript sau, trả lời câu hỏi của người dùng bằng tiếng Việt.Phần tóm tắt nội dung thì nên tóm tắt trong 3 câu, 
dựa vào các đoạn transcript được cung cấp và chỉ ra đoạn video chứa thông tin đó (video url, thời điểm bắt đầu và kết thúc).
Đồng thời làm mượt lại nội dung tóm tắt đó
Nếu không biết câu trả lời thì cứ trả lời là không biết
Định dạng đầu ra phải tuân theo JSON schema sau:
{format_instructions}
Transcript:
{context}

Câu hỏi: {question}
\nAnswer:                                          
""")

def format_doc(docs):
    return "\n\n".join(f"""
[Video URL]: {doc.metadata.get('video_url')}
[Start]: {doc.metadata.get('start_timestamp')}
[Content]: {doc.page_content}
""" for doc in docs)

def extract_json_from_output(output: str) -> str:
    return output.split('Answer')[1].strip()
    
# ===== Tạo RAG chain =====
rag_chain = (
    {
        "context": retriever | format_doc,  # format từng doc
        "question": RunnablePassthrough()
    }
    | prompt.partial(format_instructions=parser.get_format_instructions())
    | llm |
    extract_json_from_output # lấy phần đằng sau answer ( là định dạng json đã chuẩn bị)
    #RunnableLambda(lambda x: extract_json_from_output(x.content))
    |parser
)

if __name__ == "__main__":
    a = time.time()
    print("hello")
    b = time.time()
    print("Thời gian cho câu hello là ", b - a)
    start = time.time()
    query = "nhờ đạo hàm riêng bậc 1 thỉ chúng ta có thể tính được gì"
    result = rag_chain.invoke(query)

    print(result)
    end = time.time()
    print(f"Time taken: {end - start} seconds")