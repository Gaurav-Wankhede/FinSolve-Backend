from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import os
from datetime import datetime
from typing import List, Dict, Any

load_dotenv()

# MongoDB setup
MONGO_URI = os.getenv("MONGO_URI")
client = AsyncIOMotorClient(MONGO_URI)
db = client["finsolve-db"]

def get_user_collection():
    return db["users"]

def get_role_collection():
    return db["roles"]

def get_document_collection():
    return db["documents"]

def get_chat_history_collection():
    return db["chat_history"]

async def save_chat_message(username: str, role: str, query: str, response: str, model: str, source_docs: List[Dict[str, Any]]):
    """Save a chat message to the history collection"""
    history_collection = get_chat_history_collection()
    chat_entry = {
        "username": username,
        "user_role": role,
        "query": query,
        "response": response,
        "model": model,
        "source_documents": source_docs,
        "timestamp": datetime.utcnow()
    }
    
    await history_collection.insert_one(chat_entry)
    return chat_entry

async def get_user_chat_history(username: str, limit: int = 10):
    """Get the chat history for a specific user"""
    history_collection = get_chat_history_collection()
    cursor = history_collection.find({"username": username}).sort("timestamp", -1).limit(limit)
    
    chat_history = []
    async for doc in cursor:
        # Convert ObjectId to string for JSON serialization
        doc["_id"] = str(doc["_id"])
        chat_history.append(doc)
    
    # Return in chronological order (oldest first)
    return list(reversed(chat_history))
