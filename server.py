"""
FastAPI Backend Server for RAG Q&A System
- Handles RAG inference via /chat endpoint
- Stores conversation history in MongoDB
- Supports ~50 concurrent users
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import os
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
import logging

# RAG agent import
from rag.lang_graph_rag import call_agent

# ============ CONFIGURATION ============
MONGODB_URL = os.getenv("mongodb_url", "mongodb://localhost:27017")
DATABASE_NAME = "puq_qa_bot"
COLLECTION_NAME = "conversations"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="PUQ Q&A Backend", version="1.0.0")

# CORS middleware (allow Streamlit frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# MongoDB client (initialized on startup)
mongo_client: Optional[AsyncIOMotorClient] = None
db = None
conversations_collection = None

# ============ MODELS ============
class Message(BaseModel):
    role: str
    content: Any  # can be str or dict (response object)

class ChatRequest(BaseModel):
    conversation_id: str
    messages: List[Message]
    user_message: str

class ChatResponse(BaseModel):
    conversation_id: str
    response: Any
    updated_at: str

class ConversationCreate(BaseModel):
    title: str = "Cuộc trò chuyện mới"

class ConversationResponse(BaseModel):
    id: str
    title: str
    messages: List[Message]
    created_at: str
    updated_at: str

class ConversationListItem(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

# ============ DATABASE LIFECYCLE ============
@app.on_event("startup")
async def startup_db():
    global mongo_client, db, conversations_collection
    try:
        mongo_client = AsyncIOMotorClient(MONGODB_URL)
        db = mongo_client[DATABASE_NAME]
        conversations_collection = db[COLLECTION_NAME]
        # Create index on conversation_id for fast lookup
        await conversations_collection.create_index("conversation_id", unique=True)
        logger.info("✅ Connected to MongoDB")
    except Exception as e:
        logger.error(f"❌ MongoDB connection failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_db():
    if mongo_client:
        mongo_client.close()
        logger.info("🔌 MongoDB connection closed")

# ============ UTILITY FUNCTIONS ============
def serialize_mongo_doc(doc: dict) -> dict:
    """Convert MongoDB ObjectId to string"""
    if doc and "_id" in doc:
        doc["_id"] = str(doc["_id"])
    return doc

# ============ API ENDPOINTS ============

@app.get("/")
async def root():
    return {"message": "PUQ Q&A Backend API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Ping MongoDB
        await mongo_client.admin.command('ping')
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Database unhealthy: {str(e)}")

@app.post("/conversations", response_model=ConversationResponse)
async def create_conversation(convo: ConversationCreate):
    """Create a new conversation"""
    conversation_id = f"chat_{int(datetime.now().timestamp())}"
    now = datetime.now().isoformat()
    
    doc = {
        "conversation_id": conversation_id,
        "title": convo.title,
        "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
        "created_at": now,
        "updated_at": now
    }
    
    try:
        result = await conversations_collection.insert_one(doc)
        doc["_id"] = str(result.inserted_id)
        return ConversationResponse(
            id=conversation_id,
            title=doc["title"],
            messages=doc["messages"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"]
        )
    except Exception as e:
        logger.error(f"Create conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations", response_model=List[ConversationListItem])
async def list_conversations(search: Optional[str] = None):
    """List all conversations (with optional search)"""
    try:
        query = {}
        if search:
            query["title"] = {"$regex": search, "$options": "i"}
        
        cursor = conversations_collection.find(query).sort("updated_at", -1)
        convos = await cursor.to_list(length=100)
        
        return [
            ConversationListItem(
                id=c["conversation_id"],
                title=c["title"],
                created_at=c["created_at"],
                updated_at=c["updated_at"],
                message_count=len(c.get("messages", []))
            )
            for c in convos
        ]
    except Exception as e:
        logger.error(f"List conversations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conversations/{conversation_id}", response_model=ConversationResponse)
async def get_conversation(conversation_id: str):
    """Get a specific conversation with full message history"""
    try:
        doc = await conversations_collection.find_one({"conversation_id": conversation_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return ConversationResponse(
            id=doc["conversation_id"],
            title=doc["title"],
            messages=doc["messages"],
            created_at=doc["created_at"],
            updated_at=doc["updated_at"]
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint - invokes RAG and saves to MongoDB
    """
    try:
        # 1. Retrieve conversation from DB
        doc = await conversations_collection.find_one({"conversation_id": request.conversation_id})
        if not doc:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # 2. Append user message to history
        messages = doc.get("messages", [])
        messages.append({"role": "user", "content": request.user_message})
        
        # 3. Prepare chat history for agent (convert content to plain text if needed)
        chat_history = []
        for m in messages:
            content = m["content"]
            if isinstance(content, dict):
                # Extract text field from response dict
                content = content.get("text", str(content))
            chat_history.append({"role": m["role"], "content": content})
        
        # 4. Call RAG agent (synchronous call - consider making async if possible)
        response = call_agent(chat_history)
        
        # 5. Append assistant response
        messages.append({"role": "assistant", "content": response})
        
        # 6. Update title if it's still default
        title = doc["title"]
        if title == "Cuộc trò chuyện mới" and len(messages) > 1:
            # Use first user message as title (truncated)
            first_user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
            if isinstance(first_user_msg, str):
                title = first_user_msg[:35] + ("..." if len(first_user_msg) > 35 else "")
        
        # 7. Save to MongoDB
        now = datetime.now().isoformat()
        await conversations_collection.update_one(
            {"conversation_id": request.conversation_id},
            {
                "$set": {
                    "title": title,
                    "messages": messages,
                    "updated_at": now
                }
            }
        )
        
        return ChatResponse(
            conversation_id=request.conversation_id,
            response=response,
            updated_at=now
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    """Delete a conversation"""
    try:
        result = await conversations_collection.delete_one({"conversation_id": conversation_id})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str):
    """Reset a conversation to initial state"""
    try:
        now = datetime.now().isoformat()
        result = await conversations_collection.update_one(
            {"conversation_id": conversation_id},
            {
                "$set": {
                    "title": "Cuộc trò chuyện mới",
                    "messages": [{"role": "assistant", "content": "Bạn muốn hỏi gì hôm nay?"}],
                    "updated_at": now
                }
            }
        )
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return {"status": "reset", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Reset conversation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ============ RUN SERVER ============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
