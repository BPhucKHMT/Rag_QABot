from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os
load_dotenv()
googleAPIKey = os.getenv("googleAPIKey")

def get_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",   # hoặc gemini-1.5-pro, gemini-2.0-flash
        temperature=0.1,
        google_api_key=googleAPIKey,
        streaming=True  
    )
    return llm