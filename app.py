import streamlit as st
import time
import json

# ============ BACKEND AGENT ============
from rag.lang_graph_rag import call_agent  # hàm backend trả về JSON
# ========================================

# --- 1. Thiết lập giao diện ---
st.set_page_config(page_title="Chatbot xịn", layout="wide")
st.title("🤖 NVTiep Q&A")

# --- 2. Tiện ích ---
def truncate_text(text, max_length=35):
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

# --- 3. Callback Sidebar ---
def set_current_conversation(convo_id):
    st.session_state.current_conversation_id = convo_id

def create_new_conversation():
    convo_id = f"chat_{int(time.time())}"
    st.session_state.conversations[convo_id] = {
        "title": "Cuộc trò chuyện mới",
        "messages": [
            {"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}
        ]
    }
    st.session_state.current_conversation_id = convo_id

# --- 4. Khởi tạo Session State ---
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

if not st.session_state.conversations:
    create_new_conversation()

# --- 5. Sidebar ---
with st.sidebar:
    st.title("Cuộc trò chuyện")

    st.button("➕ Cuộc trò chuyện mới",
              on_click=create_new_conversation,
              use_container_width=True)

    st.divider()
    st.subheader("Gần đây")

    convo_ids = list(st.session_state.conversations.keys())
    for convo_id in reversed(convo_ids):
        title = st.session_state.conversations[convo_id]["title"]
        is_active = (convo_id == st.session_state.current_conversation_id)

        st.button(
            title,
            on_click=set_current_conversation,
            args=(convo_id,),
            use_container_width=True,
            type="primary" if is_active else "secondary"
        )

# --- 6. Khu vực Chat Chính ---
current_id = st.session_state.current_conversation_id

if current_id and current_id in st.session_state.conversations:

    current_convo = st.session_state.conversations[current_id]
    messages = current_convo["messages"]

    # 1. Hiển thị lịch sử chat
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Input người dùng
    if prompt := st.chat_input("Nhắn tin..."):

        # Lưu tin nhắn user
        messages.append({"role": "user", "content": prompt})

        # Cập nhật title nếu là chat mới
        rerun_needed = False
        if current_convo["title"] == "Cuộc trò chuyện mới":
            current_convo["title"] = truncate_text(prompt)
            rerun_needed = True

        # Hiển thị cho người dùng
        with st.chat_message("user"):
            st.markdown(prompt)

        # Chuẩn hóa lịch sử chat cho backend
        chat_history = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
        ]

        # --- Gọi backend agent ---
        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."):
                raw_response = call_agent(chat_history)

                # Parse JSON trả về
                try:
                    answer = raw_response
                except json.JSONDecodeError:
                    st.markdown(raw_response)
                    answer = None

                if answer:
                    # Hiển thị Summary
                    st.markdown(f"**Summary:**\n{answer['text']}")

                    st.divider()
                    st.markdown("**Videos tham khảo:**")
                    for url, start, end in zip(answer['video_url'], answer['start_timestamp'], answer['end_timestamp']):
                        st.markdown(f"- [{url}]({url}) (Từ {start} đến {end})")

                    # Lưu full response vào messages (summary + video list)
                    video_refs = "\n".join([f"{u} (Từ {s} đến {e})"
                                            for u, s, e in zip(answer['video_url'], answer['start_timestamp'], answer['end_timestamp'])])
                    full_response = f"{answer['text']}\n\nVideos tham khảo:\n{video_refs}"
                    messages.append({"role": "assistant", "content": full_response})
                else:
                    # Nếu không parse được, lưu raw_response
                    messages.append({"role": "assistant", "content": raw_response})

        # Rerun để update sidebar (nếu đổi title)
        if rerun_needed:
            st.rerun()

else:
    st.info("Vui lòng tạo hoặc chọn một cuộc trò chuyện từ thanh bên.")
