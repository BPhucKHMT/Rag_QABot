# Hướng Dẫn Deploy PUQ Q&A với Backend/Frontend Tách Biệt

Hệ thống này đã được tách thành:
- **Backend**: `server.py` (FastAPI + MongoDB) - xử lý RAG và lưu lịch sử chat
- **Frontend**: `app.py` (Streamlit) - giao diện người dùng

## 🔧 Yêu Cầu Hệ Thống

- Python 3.10+
- MongoDB (local hoặc MongoDB Atlas)
- CUDA-compatible GPU (optional, cho embedding models)

## 📦 Cài Đặt Dependencies

```powershell
# Clone repo (nếu chưa có)
git clone <your-repo-url>
cd final_project

# Tạo virtual environment (khuyên dùng)
conda create -n puq_qa python=3.10
conda activate puq_qa

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🗄️ Setup MongoDB

### Option 1: MongoDB Local (Windows)
```powershell
# Download và cài MongoDB Community Server từ:
# https://www.mongodb.com/try/download/community

# Start MongoDB service
net start MongoDB

# MongoDB sẽ chạy ở mongodb://localhost:27017
```

### Option 2: MongoDB Atlas (Cloud - Miễn phí 512MB)
1. Tạo tài khoản tại https://www.mongodb.com/cloud/atlas
2. Tạo cluster miễn phí
3. Lấy connection string (dạng: `mongodb+srv://user:pass@cluster.mongodb.net/`)
4. Whitelist IP của bạn (hoặc cho phép 0.0.0.0/0 nếu test)

## 🚀 Chạy Hệ Thống

### Bước 1: Cấu hình MongoDB Connection

**Nếu dùng MongoDB local:**
```powershell
# Windows PowerShell
$env:MONGODB_URL = "mongodb://localhost:27017"
```

**Nếu dùng MongoDB Atlas:**
```powershell
# Thay YOUR_CONNECTION_STRING bằng string từ Atlas
$env:MONGODB_URL = "mongodb+srv://user:pass@cluster.mongodb.net/"
```

**Hoặc tạo file `.env`:**
```bash
MONGODB_URL=mongodb://localhost:27017
# hoặc
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/
```

### Bước 2: Chạy Backend Server

```powershell
# Terminal 1 - Chạy FastAPI backend
python server.py

# Hoặc dùng uvicorn với reload (dev mode)
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

Backend sẽ chạy ở: **http://localhost:8000**

API docs (Swagger UI): **http://localhost:8000/docs**

### Bước 3: Chạy Streamlit Frontend

```powershell
# Terminal 2 - Chạy Streamlit frontend
streamlit run app.py
```

Frontend sẽ mở tự động ở: **http://localhost:8501**

## 🔍 Kiểm Tra Hệ Thống

### Test Backend Health
```powershell
# PowerShell
curl http://localhost:8000/health
```

Output mong đợi:
```json
{"status":"healthy","database":"connected"}
```

### Test API Endpoints
```powershell
# Tạo conversation mới
curl -X POST http://localhost:8000/conversations -H "Content-Type: application/json" -d '{"title":"Test"}'

# List conversations
curl http://localhost:8000/conversations
```

## 🌐 Deploy Production (Cho ~50 Users)

### Option A: Deploy Trên VPS/Cloud VM

**Yêu cầu VPS:**
- 4GB RAM (minimum)
- 2 vCPU
- 20GB disk
- Ubuntu 20.04+ hoặc Windows Server

**Setup steps:**
```bash
# 1. Clone code lên server
git clone <repo>
cd final_project

# 2. Cài Python và dependencies
pip install -r requirements.txt

# 3. Cài MongoDB hoặc dùng Atlas
# (xem phần MongoDB setup ở trên)

# 4. Chạy backend với gunicorn (production server)
pip install gunicorn
gunicorn server:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# 5. Chạy Streamlit
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

**Setup Nginx Reverse Proxy (recommended):**
```nginx
# /etc/nginx/sites-available/puq_qa
server {
    listen 80;
    server_name your-domain.com;

    # Backend API
    location /api/ {
        proxy_pass http://localhost:8000/;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    # Frontend
    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}
```

### Option B: Deploy Với Docker (Recommended)

Tạo `docker-compose.yml`:
```yaml
version: '3.8'
services:
  mongodb:
    image: mongo:7
    restart: always
    ports:
      - "27017:27017"
    volumes:
      - mongo_data:/data/db

  backend:
    build: .
    command: uvicorn server:app --host 0.0.0.0 --port 8000
    environment:
      - MONGODB_URL=mongodb://mongodb:27017
    ports:
      - "8000:8000"
    depends_on:
      - mongodb

  frontend:
    build: .
    command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
    environment:
      - API_BASE_URL=http://backend:8000
    ports:
      - "8501:8501"
    depends_on:
      - backend

volumes:
  mongo_data:
```

Chạy:
```bash
docker-compose up -d
```

## 📊 Monitoring & Scaling

### Giám sát Backend
- Check logs: `tail -f server.log`
- Monitor MongoDB: `mongosh` -> `db.conversations.stats()`
- API metrics: thêm prometheus/grafana nếu cần

### Scale cho 50+ users
- **Backend**: tăng số workers trong gunicorn (4-8 workers cho 50 users)
- **MongoDB**: dùng replica set hoặc sharding nếu > 10GB data
- **Frontend**: deploy nhiều instance Streamlit + load balancer
- **Caching**: thêm Redis để cache RAG responses

## 🐛 Troubleshooting

### Lỗi: "Cannot connect to MongoDB"
```powershell
# Kiểm tra MongoDB đang chạy
# Windows:
Get-Service MongoDB

# Linux:
systemctl status mongod
```

### Lỗi: "Connection refused localhost:8000"
- Đảm bảo backend đang chạy: `curl http://localhost:8000/health`
- Check port conflict: `netstat -ano | findstr :8000`

### Lỗi: Frontend không gọi được API
- Sửa `API_BASE_URL` trong `app.py` (nếu backend ở domain khác)
- Check CORS settings trong `server.py`

## 📝 API Endpoints Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root endpoint |
| GET | `/health` | Health check |
| POST | `/conversations` | Tạo conversation mới |
| GET | `/conversations` | List conversations |
| GET | `/conversations/{id}` | Get conversation detail |
| POST | `/chat` | Gửi message & nhận response |
| DELETE | `/conversations/{id}` | Xóa conversation |
| POST | `/conversations/{id}/reset` | Reset conversation |

Full API docs: http://localhost:8000/docs

## 🔐 Security Notes (Production)

- [ ] Đặt CORS `allow_origins` cụ thể (không dùng `*`)
- [ ] Dùng HTTPS (Let's Encrypt)
- [ ] Đặt MongoDB authentication
- [ ] Rate limiting (dùng slowapi)
- [ ] API key authentication (thêm middleware)

## 💡 Tips

- Dùng MongoDB Atlas free tier (512MB) cho testing
- Monitor RAM usage: backend + embedding models cần ~4GB
- Set `max_connections` trong MongoDB nếu nhiều users
- Dùng async/await cho I/O-bound operations

---

**Liên hệ**: Nếu gặp vấn đề, tạo issue trên GitHub hoặc liên hệ admin.
