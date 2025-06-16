from motor.motor_asyncio import AsyncIOMotorClient
import os
from datetime import datetime
from typing import List, Dict, Any
import logging

# Set up logging
logger = logging.getLogger(__name__)

# MongoDB setup for serverless environment
def get_mongodb_client():
    """Get a fresh MongoDB client for each request"""
    try:
        MONGO_URI = os.environ.get("MONGO_URI")
        if not MONGO_URI:
            logger.error("MONGO_URI environment variable is not set")
            raise ValueError("MONGO_URI environment variable is not set")
            
        # Create a new client for each request
        client = AsyncIOMotorClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {str(e)}")
        raise

def get_database():
    """Get database connection"""
    client = get_mongodb_client()
    return client["finsolve-db"]

def get_user_collection():
    """Get user collection with fresh connection"""
    db = get_database()
    return db["users"]

def get_role_collection():
    """Get role collection with fresh connection"""
    db = get_database()
    return db["roles"]

def get_document_collection():
    """Get document collection with fresh connection"""
    db = get_database()
    return db["documents"]

def get_chat_history_collection():
    """Get chat history collection with fresh connection"""
    db = get_database()
    return db["chat_history"]

async def save_chat_message(username: str, role: str, query: str, response: str, model: str, source_docs: List[Dict[str, Any]]):
    """Save a chat message to the history collection"""
    try:
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
    except Exception as e:
        logger.error(f"Error saving chat message: {str(e)}")
        raise

async def get_user_chat_history(username: str, limit: int = 10):
    """Get the chat history for a specific user"""
    try:
        history_collection = get_chat_history_collection()
        cursor = history_collection.find({"username": username}).sort("timestamp", -1).limit(limit)
        
        chat_history = []
        async for doc in cursor:
            # Convert ObjectId to string for JSON serialization
            doc["_id"] = str(doc["_id"])
            chat_history.append(doc)
        
        # Return in chronological order (oldest first)
        return list(reversed(chat_history))
    except Exception as e:
        logger.error(f"Error getting chat history: {str(e)}")
        return []