from fastapi import APIRouter, Depends, HTTPException
from utils.mongo import get_document_collection
from services.chatbot import get_current_user
from bson import ObjectId

debug_router = APIRouter()

@debug_router.get("/debug/documents", tags=["debug"])
async def debug_document_access(current_user: dict = Depends(get_current_user)):
    """Debug endpoint to check document access for the current user's role"""
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can access debug endpoints")
    
    try:
        documents = get_document_collection()
        
        # Get all documents
        all_docs = []
        async for doc in documents.find({}, {"_id": 1, "title": 1, "category": 1, "allowed_roles": 1}):
            doc["_id"] = str(doc["_id"])
            all_docs.append(doc)
            
        # Get documents for the current role
        role_docs = []
        async for doc in documents.find({"allowed_roles": current_user["role"]}, {"_id": 1, "title": 1, "category": 1, "allowed_roles": 1}):
            doc["_id"] = str(doc["_id"])
            role_docs.append(doc)
            
        return {
            "user_role": current_user["role"],
            "total_documents": len(all_docs),
            "role_accessible_documents": len(role_docs),
            "all_documents": all_docs,
            "role_documents": role_docs
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error debugging documents: {str(e)}")

@debug_router.get("/debug/documents/{document_id}/embeddings", tags=["debug"])
async def debug_document_embeddings(document_id: str, current_user: dict = Depends(get_current_user)):
    """Debug endpoint to check embeddings for a specific document"""
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can access debug endpoints")
    
    try:
        documents = get_document_collection()
        doc = await documents.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        has_embeddings = "embedding_chunks" in doc and doc["embedding_chunks"]
        embedding_count = len(doc.get("embedding_chunks", [])) if has_embeddings else 0
        embedding_sample = doc["embedding_chunks"][0][:10] if embedding_count > 0 else None
            
        return {
            "document_id": str(doc["_id"]),
            "title": doc.get("title", "Untitled"),
            "has_embeddings": has_embeddings,
            "embedding_chunks_count": embedding_count,
            "embedding_sample": embedding_sample
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error debugging embeddings: {str(e)}") 