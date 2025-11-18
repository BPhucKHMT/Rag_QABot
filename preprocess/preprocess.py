from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

googleAPIKey = os.getenv('googleAPIKey')
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",   # hoặc gemini-1.5-pro, gemini-2.0-flash
    temperature=0.0,
    google_api_key=googleAPIKey  # 👈 thêm dòng này
)

# Hàm sửa chính tả
def correct_spelling(text):
    prompt = f"""
    Bạn nhận được transcript gồm nhiều dòng, mỗi dòng có timestamp dạng "HH:MM:SS - HH:MM:SS, nội dung".
    Hãy làm đúng theo hai quy tắc, không làm thêm gì khác:

    1. Chuẩn hóa timestamp:
       - Bỏ số 0 dư ở đầu giờ, ví dụ "00:00:05" -> "0:00:05".
       - Thay những đoạn có định dạng từ 00:00:00 hoặc  0:00:00 thành 0:00:14.
       - Không thay đổi phút, giây hoặc dấu phẩy.
       - Giữ nguyên mọi timestamp, không thêm hay xóa dòng.

    2. Sửa chính tả:
       - Chỉ sửa lỗi chính tả trong phần văn bản sau dấu phẩy.
       - Giữ nguyên dấu câu, khoảng trắng, chữ hoa/thường, cấu trúc câu.
       - Không thêm, xóa, hoặc di chuyển bất kỳ ký tự nào ngoài sửa chính tả.

    Transcript:
    {text}

    Chỉ trả về kết quả transcript đã chỉnh sửa, không giải thích.
    """
    response = llm.invoke(prompt)
    return response.content
