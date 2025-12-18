
import streamlit as st
import time
import json
import re
import requests
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

# ============ CONFIGURATION ============
API_BASE_URL = "http://localhost:8000"  # FastAPI backend
CONVERSATIONS_DIR = "saved_conversations"  # Not used anymore (using MongoDB via API)
Path(CONVERSATIONS_DIR).mkdir(exist_ok=True)  # Keep for backward compatibility

# ============ UTILITIES ============
def truncate_text(text, max_length=35):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

def timestamp_to_seconds(timestamp: str) -> int:
    """Chuyển HH:MM:SS hoặc MM:SS sang seconds"""
    try:
        parts = list(map(int, timestamp.split(':')))
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
        elif len(parts) == 2:
            return parts[0] * 60 + parts[1]
    except:
        pass
    return 0

def response_to_display_text(response) -> str:
    """Convert response thành plain text"""
    if isinstance(response, dict):
        text = response.get('text', '')
        clean_text = re.sub(r'\[(\d+)\]', r'[\1]', text)
        return clean_text
    elif isinstance(response, str):
        return response
    else:
        return str(response)

def render_response(response):
    """Universal renderer"""
    if isinstance(response, dict):
        response_type = response.get('type', 'unknown')
        text = response.get('text', '')
        video_urls = response.get('video_url', [])
        titles = response.get('title', [])
        start_timestamps = response.get('start_timestamp', [])
        end_timestamps = response.get('end_timestamp', [])
        confidences = response.get('confidence', [])
        
        if video_urls:
            def replace_citation(match):
                index = int(match.group(1))
                if index < len(video_urls):
                    url = video_urls[index]
                    title = titles[index] if index < len(titles) else f"Video {index}"
                    start = start_timestamps[index] if index < len(start_timestamps) else "00:00:00"
                    seconds = timestamp_to_seconds(start)
                    video_link = f"{url}&t={seconds}" if '?' in url else f"{url}?t={seconds}"
                    return f'<a href="{video_link}" target="_blank" style="color: #1E88E5; font-weight: bold; text-decoration: none; border-bottom: 1px dotted #1E88E5;" title="{title} - {start}">[{index}]</a>'
                return match.group(0)
            formatted_text = re.sub(r'\[(\d+)\]', replace_citation, text)
        else:
            formatted_text = text
        
        st.markdown(formatted_text, unsafe_allow_html=True)
        
        if video_urls and response_type == "rag":
            st.markdown("---")
            st.markdown("### 📺 Nguồn tham khảo:")
            for i, url in enumerate(video_urls):
                title = titles[i] if i < len(titles) else f"Video {i}"
                start = start_timestamps[i] if i < len(start_timestamps) else "00:00:00"
                end = end_timestamps[i] if i < len(end_timestamps) else start
                confidence = confidences[i] if i < len(confidences) else "unknown"
                seconds = timestamp_to_seconds(start)
                video_link = f"{url}&t={seconds}" if '?' in url else f"{url}?t={seconds}"
                conf_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🟠', 'zero': '🔴'}.get(confidence, '⚪')
                st.markdown(f"**{i}.** {conf_emoji} [{title}]({video_link}) ⏱️ `{start}` → `{end}`")
    
    elif isinstance(response, str):
        st.markdown(response, unsafe_allow_html=True)
    else:
        st.error(f"⚠️ Unknown response format: {type(response)}")

# ============ API CLIENT FUNCTIONS ============
def api_request(method: str, endpoint: str, **kwargs) -> Any:
    """Generic API request wrapper with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        response = requests.request(method, url, timeout=60, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def load_all_conversations():
    """Load tất cả conversations từ API (MongoDB)"""
    try:
        data = api_request("GET", "/conversations")
        if not data:
            return {}
        
        conversations = {}
        for item in data:
            conversations[item["id"]] = {
                "title": item["title"],
                "messages": [],  # Will be loaded on demand
                "created_at": item["created_at"],
                "updated_at": item["updated_at"]
            }
        return conversations
    except Exception as e:
        st.error(f"Failed to load conversations: {e}")
        return {}

def load_conversation_messages(convo_id: str) -> List[Dict]:
    """Load full message history for a conversation"""
    try:
        data = api_request("GET", f"/conversations/{convo_id}")
        if data:
            return data.get("messages", [])
        return []
    except Exception as e:
        st.error(f"Failed to load messages: {e}")
        return []

def delete_conversation(convo_id: str):
    """Xóa conversation via API"""
    try:
        result = api_request("DELETE", f"/conversations/{convo_id}")
        if result:
            # Xóa khỏi session state
            if convo_id in st.session_state.conversations:
                del st.session_state.conversations[convo_id]
        
        # Reset current ID nếu đang active
        if st.session_state.current_conversation_id == convo_id:
            remaining_convos = list(st.session_state.conversations.keys())
            if remaining_convos:
                st.session_state.current_conversation_id = remaining_convos[-1]
            else:
                create_new_conversation()
        
        return True
    except:
        return False

def reset_conversation(convo_id: str):
    """Reset conversation via API"""
    try:
        result = api_request("POST", f"/conversations/{convo_id}/reset")
        if result:
            # Update local state
            st.session_state.conversations[convo_id] = {
                "title": "Cuộc trò chuyện mới",
                "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
                "created_at": datetime.now().isoformat()
            }
            return True
        return False
    except:
        return False

# ============ SESSION MANAGEMENT ============
def create_new_conversation():
    """Tạo conversation mới via API"""
    try:
        result = api_request("POST", "/conversations", json={"title": "Cuộc trò chuyện mới"})
        if result:
            convo_id = result["id"]
            st.session_state.conversations[convo_id] = {
                "title": result["title"],
                "messages": result["messages"],
                "created_at": result["created_at"]
            }
            st.session_state.current_conversation_id = convo_id
        else:
            st.error("Failed to create conversation")
    except Exception as e:
        st.error(f"Error creating conversation: {e}")

def set_current_conversation(convo_id):
    """Switch conversation"""
    st.session_state.current_conversation_id = convo_id

# ============ SETUP PAGE ============
st.set_page_config(
    page_title="PUQ Q&A",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stButton button {
        width: 100%;
    }
    .block-container {
        padding-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🤖 PUQ Q&A")

# ============ INITIALIZE SESSION STATE ============
if "conversations" not in st.session_state:
    # Load từ disk
    st.session_state.conversations = load_all_conversations()

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if not st.session_state.conversations:
    create_new_conversation()

# ============ SIDEBAR ============
with st.sidebar:
    st.title("💬 Cuộc trò chuyện")
    
    # New conversation button
    if st.button("➕ Cuộc trò chuyện mới", use_container_width=True):
        create_new_conversation()
        st.rerun()
    
    st.divider()
    
    # Search box
    search_query = st.text_input("🔍 Tìm kiếm", placeholder="Nhập từ khóa...")
    
    st.subheader("Gần đây")
    
    convo_ids = list(st.session_state.conversations.keys())
    
    # Filter by search
    if search_query:
        filtered_ids = [
            cid for cid in convo_ids
            if search_query.lower() in st.session_state.conversations[cid]["title"].lower()
        ]
    else:
        filtered_ids = convo_ids
    
    # Display conversations
    for convo_id in reversed(filtered_ids):
        convo = st.session_state.conversations[convo_id]
        title = convo["title"]
        is_active = (convo_id == st.session_state.current_conversation_id)
        
        # Conversation item với delete/reset
        col1, col2, col3 = st.columns([7, 1.5, 1.5])
        
        with col1:
            if st.button(
                title,
                key=f"select_{convo_id}",
                type="primary" if is_active else "secondary",
                use_container_width=True
            ):
                set_current_conversation(convo_id)
                st.rerun()
        
        with col2:
            if st.button("🗑️", key=f"delete_{convo_id}", help="Xóa"):
                if delete_conversation(convo_id):
                    st.rerun()
        
        with col3:
            if st.button("🔄", key=f"reset_{convo_id}", help="Reset"):
                if reset_conversation(convo_id):
                    st.rerun()

# ============ MAIN CHAT AREA ============
current_id = st.session_state.current_conversation_id

if current_id and current_id in st.session_state.conversations:
    current_convo = st.session_state.conversations[current_id]
    # Load full messages from API if not already loaded
    if not current_convo.get("messages") or len(current_convo["messages"]) == 0:
        current_convo["messages"] = load_conversation_messages(current_id)
    messages = current_convo["messages"]
    
    # Display messages
    for message in messages:
        with st.chat_message(message["role"]):
            content = message["content"]
            if message["role"] == "assistant":
                render_response(content)
            else:
                st.markdown(content)
    
    # User input
    if prompt := st.chat_input("Nhắn tin..."):
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        # Update title if new conversation
        if current_convo["title"] == "Cuộc trò chuyện mới":
            current_convo["title"] = truncate_text(prompt)
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Prepare chat history
        chat_history = []
        for m in messages:
            content = m["content"]
            if isinstance(content, dict):
                content = response_to_display_text(content)
            chat_history.append({"role": m["role"], "content": content})
        
        # Call backend API for RAG response
        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."):
                try:
                    # Call /chat endpoint
                    result = api_request(
                        "POST",
                        "/chat",
                        json={
                            "conversation_id": current_id,
                            "messages": chat_history,
                            "user_message": prompt
                        }
                    )
                    
                    if result:
                        response = result["response"]
                        render_response(response)
                        messages.append({"role": "assistant", "content": response})
                        
                        # Update title if changed
                        if current_convo["title"] == "Cuộc trò chuyện mới":
                            current_convo["title"] = truncate_text(prompt)
                    else:
                        error_msg = "⚠️ Không thể kết nối với server"
                        st.error(error_msg)
                        messages.append({"role": "assistant", "content": error_msg})
                    
                except Exception as e:
                    error_msg = f"⚠️ Có lỗi xảy ra: {str(e)}"
                    st.error(error_msg)
                    messages.append({"role": "assistant", "content": error_msg})
        
        st.rerun()

else:
    st.info("👈 Vui lòng chọn hoặc tạo cuộc trò chuyện từ thanh bên.")