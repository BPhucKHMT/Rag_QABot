from dotenv import load_dotenv
import os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()

googleAPIKey = os.getenv("googleAPIKey")
from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # hoặc gemini-1.5-pro, gemini-2.0-flash
    temperature=0.0,
    google_api_key=googleAPIKey,  # 👈 thêm dòng này
)


# Hàm sửa chính tả
def correct_spelling(text):
    prompt = f"""
    Chỉ sửa các từ sai chính tả trong đoạn sau, giữ nguyên tất cả dấu câu, số, khoảng trắng và cấu trúc. 
    Không thêm hoặc xóa bất kỳ câu nào. Trả về nguyên văn sau khi sửa:
    
    {text}
    """
    response = llm.invoke(prompt)
    return response.conten
