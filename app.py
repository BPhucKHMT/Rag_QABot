import streamlit as st
import time

# --- 1. Thiết lập Trang và Tiêu đề ---
st.set_page_config(page_title="Chatbot xịn", layout="wide")
st.title("🤖 NVTiep Q&A")

# --- 2. Hàm Tiện ích ---

def truncate_text(text, max_length=35):
    """Một hàm nhỏ để cắt ngắn tiêu đề cho sidebar"""
    if len(text) > max_length:
        return text[:max_length] + "..."
    return text

# --- 3. Định nghĩa các hàm Callback ---

def set_current_conversation(convo_id):
    """Cập nhật session_state để chọn một cuộc trò chuyện"""
    st.session_state.current_conversation_id = convo_id

def create_new_conversation():
    """
    Tạo một cuộc trò chuyện mới với ID duy nhất (dựa trên timestamp)
    và chọn nó làm cuộc trò chuyện hiện tại.
    """
    # Sử dụng timestamp làm ID duy nhất và để sắp xếp
    convo_id = f"chat_{int(time.time())}" 
    
    # Cấu trúc dữ liệu mới:
    # Mỗi cuộc trò chuyện là một dict chứa 'title' và 'messages'
    st.session_state.conversations[convo_id] = {
        "title": "Cuộc trò chuyện mới", # Tiêu đề mặc định
        "messages": [
            {"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}
        ]
    }
    # Chọn cuộc trò chuyện mới này
    st.session_state.current_conversation_id = convo_id

# --- 4. Khởi tạo Session State ---

# 'conversations' là một TỪ ĐIỂN (dict)
# key: ID duy nhất (ví dụ: "chat_1678886400")
# value: một dict khác chứa { "title": "...", "messages": [...] }
if "conversations" not in st.session_state:
    st.session_state.conversations = {}

# 'current_conversation_id' theo dõi ID của cuộc trò chuyện đang xem
if "current_conversation_id" not in st.session_state:
    st.session_state.current_conversation_id = None

# Tự động tạo và chọn cuộc trò chuyện đầu tiên nếu chưa có
if not st.session_state.conversations:
    create_new_conversation()

# --- 5. Tạo Thanh bên (Sidebar) ---

with st.sidebar:
    st.title("Cuộc trò chuyện")
    
    # Nút "Cuộc trò chuyện mới"
    st.button("➕ Cuộc trò chuyện mới", 
              on_click=create_new_conversation, 
              use_container_width=True)
    
    st.divider() 
    st.subheader("Gần đây") # Giống như trong ảnh của bạn

    # Hiển thị danh sách các cuộc trò chuyện
    # Sắp xếp theo ID (timestamp) để cuộc trò chuyện mới nhất lên đầu
    # Chúng ta đảo ngược (reversed) danh sách keys
    convo_ids = list(st.session_state.conversations.keys())
    for convo_id in reversed(convo_ids):
        
        # Lấy tiêu đề hiển thị từ cấu trúc dữ liệu
        display_title = st.session_state.conversations[convo_id]["title"]
        
        is_active = (convo_id == st.session_state.current_conversation_id)
        
        st.button(
            display_title,
            on_click=set_current_conversation,
            args=(convo_id,),
            use_container_width=True,
            type="primary" if is_active else "secondary"
        )

# --- 6. Tạo Khu vực Chat Chính ---

current_id = st.session_state.current_conversation_id

# Chỉ hiển thị nếu có một cuộc trò chuyện đang được chọn
if current_id and current_id in st.session_state.conversations:
    
    # Lấy dữ liệu của cuộc trò chuyện hiện tại
    current_convo_data = st.session_state.conversations[current_id]
    messages = current_convo_data["messages"]
    
    # 1. Hiển thị lịch sử tin nhắn
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 2. Xử lý input mới từ người dùng
    if prompt := st.chat_input(f"Nhắn tin..."):
        
        # a. Thêm tin nhắn của người dùng vào danh sách
        messages.append({"role": "user", "content": prompt})
        
        # --- LOGIC QUAN TRỌNG: CẬP NHẬT TIÊU ĐỀ ---
        # Nếu tiêu đề vẫn là mặc định ("Cuộc trò chuyện mới"),
        # cập nhật nó bằng nội dung prompt đầu tiên của người dùng.
        rerun_needed = False
        if current_convo_data["title"] == "Cuộc trò chuyện mới":
            current_convo_data["title"] = truncate_text(prompt)
            rerun_needed = True # Báo hiệu cần chạy lại để update sidebar

        # Hiển thị tin nhắn của user ngay lập tức
        with st.chat_message("user"):
            st.markdown(prompt)

        # b. Tạo phản hồi "giả" (dummy) từ bot
        response_content = f"Tôi là phản hồi cho câu: \"{prompt}\""

        # c. Hiển thị phản hồi của bot
        with st.chat_message("assistant"):
            with st.spinner("Bot đang suy nghĩ..."):
                time.sleep(1.0) # Giả lập
            st.markdown(response_content)
        
        # d. Thêm phản hồi của bot vào danh sách
        messages.append({"role": "assistant", "content": response_content})

        # Nếu chúng ta vừa cập nhật tiêu đề, hãy chạy lại script
        # để sidebar hiển thị tiêu đề mới ngay lập tức
        if rerun_needed:
            st.rerun()
else:
    st.info("Vui lòng tạo hoặc chọn một cuộc trò chuyện từ thanh bên.")