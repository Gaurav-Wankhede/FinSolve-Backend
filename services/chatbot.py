from fastapi import APIRouter, Depends, HTTPException
from typing import List
from pydantic import BaseModel
from bson import ObjectId
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Security
from jose import jwt, JWTError
import os
import numpy as np
from utils.mongo import get_document_collection, save_chat_message, get_user_chat_history
from schemas.document_models import QueryRequest, QueryResponse, DocumentMetadata
from services.google_embeddings import get_google_embedding
from services.langchain_rag import LangChainRAG
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek/deepseek-r1-0528:free")

chatbot_router = APIRouter()

security = HTTPBearer(
    scheme_name="Bearer",
    description="JWT token authentication",
    auto_error=True
)

def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("username")
        role = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Username or role not found in token")
        return {"username": username, "role": role}
    except (ValueError, JWTError) as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=401, detail="Invalid token")

def cosine_similarity(a, b):
    """Calculate cosine similarity between two vectors"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

async def semantic_search(query_embedding, documents, top_k=3):
    """Search for documents similar to the query"""
    results = []
    
    for doc in documents:
        # Check if document has chunks
        if "embedding_chunks" not in doc or not doc["embedding_chunks"]:
            continue
            
        # Find the highest similarity among chunks
        max_similarity = 0
        for chunk_embedding in doc["embedding_chunks"]:
            similarity = cosine_similarity(query_embedding, chunk_embedding)
            max_similarity = max(max_similarity, similarity)
            
        if max_similarity > 0:
            results.append({
                "document": doc,
                "similarity": max_similarity
            })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top k results
    return results[:top_k]

class ModelRequest(BaseModel):
    """Request to specify which model to use"""
    model: str = DEFAULT_MODEL  # Default to environment variable
    use_history: bool = True  # Whether to use chat history

class TestModelRequest(BaseModel):
    """Request for testing model generation"""
    prompt: str
    model: str = DEFAULT_MODEL  # Default to environment variable

class ModelInfo(BaseModel):
    """Model information"""
    id: str
    name: str
    description: str
    provider: str

@chatbot_router.post("/chat", response_model=QueryResponse, tags=["chatbot"])
async def chat(
    query: QueryRequest,
    model_request: ModelRequest = None,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Get query embedding using Google's embedding service
        query_embedding = await get_google_embedding(query.query)
        
        # Get documents collection
        documents = get_document_collection()
        
        # Fetch documents user has access to
        user_role = current_user["role"]
        username = current_user["username"]
        
        # Log information about the query
        logger.info(f"Processing query for user {username} with role {user_role}: {query.query[:50]}...")
        
        # Get total document count for debugging
        total_count = await documents.count_documents({})
        logger.info(f"Total documents in collection: {total_count}")
        
        # Get role-specific document count for debugging
        role_count = await documents.count_documents({"allowed_roles": user_role})
        logger.info(f"Documents accessible to role '{user_role}': {role_count}")
        
        # Check if any documents exist for this role
        if role_count == 0:
            logger.warning(f"No documents found for role '{user_role}'. Please upload documents with appropriate permissions.")
            return QueryResponse(
                answer=f"I'm sorry, but I don't have any information available for your role ({user_role}). Please ask an administrator to upload relevant documents.",
                source_documents=[]
            )
        
        # Fetch the actual documents
        accessible_docs = []
        async for doc in documents.find({"allowed_roles": user_role}):
            # Check if document has embeddings
            if "embedding_chunks" not in doc or not doc["embedding_chunks"]:
                logger.warning(f"Document {doc.get('_id')} missing embeddings: {doc.get('title', 'Untitled')}")
                continue
            accessible_docs.append(doc)
            
        if not accessible_docs:
            logger.warning(f"No documents with embeddings found for role '{user_role}'")
            return QueryResponse(
                answer=f"I'm sorry, but I don't have any properly indexed information available for your role ({user_role}). Please ask an administrator to check document embeddings.",
                source_documents=[]
            )
        
        logger.info(f"Found {len(accessible_docs)} documents with embeddings for role '{user_role}'")
        
        # Get model to use and history preference
        model = DEFAULT_MODEL  # Default from environment
        use_history = True
        if model_request:
            model = model_request.model
            use_history = model_request.use_history
            
        # Get chat history if enabled
        chat_history = None
        if use_history:
            chat_history = await get_user_chat_history(username, limit=5)
            
        # Initialize LangChain RAG with the selected model
        rag = LangChainRAG(model_name=model)
            
        # Perform semantic search using Google embeddings
        search_results = await rag.semantic_search(query_embedding, accessible_docs)
        
        if not search_results:
            logger.warning(f"No relevant documents found for query: {query.query[:50]}...")
            return QueryResponse(
                answer="I couldn't find any relevant information for your query. Please try rephrasing your question or ask about a different topic.",
                source_documents=[]
            )
        
        logger.info(f"Found {len(search_results)} relevant documents for the query")
        
        # Extract documents from search results
        retrieved_docs = [result["document"] for result in search_results]

        
        # Enhance CSV handling, especially for HR data
        # Special handling for CSV data in vector database
        for doc in retrieved_docs:
            content = doc.get("document", "")
            
            # Check if this might be CSV data
            if content and isinstance(content, str):
                # CSV detection based on content patterns
                if (content.count(',') > 3 and content.count('\n') > 1) or \
                (doc.get("filename", "").endswith(".csv")) or \
                (doc.get("title", "").endswith(".csv")):
                    
                    doc["format"] = "csv"
                    
                    # Try to reconstruct CSV structure for better processing
                    try:
                        lines = content.strip().split('\n')
                        if lines and ',' in lines[0]:
                            # Extract headers from first line
                            headers = [h.strip() for h in lines[0].split(',')]
                            doc["csv_headers"] = headers
                            
                            # For HR data detection
                            hr_related_terms = ["employee", "personnel", "salary", "position", 
                                            "department", "hire", "staff", "team member"]
                            
                            # Check if this is HR data
                            is_hr_data = False
                            if "hr" in doc.get("category", "").lower():
                                is_hr_data = True
                            else:
                                # Check headers for HR-related terms
                                for header in headers:
                                    if any(term in header.lower() for term in hr_related_terms):
                                        is_hr_data = True
                                        break
                            
                            if is_hr_data:
                                doc["hr_data"] = True
                    except Exception as e:
                        logger.warning(f"Error processing CSV structure: {e}")

        # Format source documents for response
        source_documents = []
        for doc in retrieved_docs:
            source_documents.append(DocumentMetadata(
                id=str(doc["_id"]),
                title=doc["title"],
                category=doc["category"]
            ))
            
        # Generate response using LangChain RAG with history
        llm_response = await rag.generate_response(
            query=query.query,
            retrieved_docs=retrieved_docs,
            user_role=user_role,
            chat_history=chat_history
        )
        
        logger.info(f"Generated response of length {len(llm_response)}")
        
        # Save chat message to history
        await save_chat_message(
            username=username,
            role=user_role,
            query=query.query,
            response=llm_response,
            model=model,
            source_docs=[{
                "id": str(doc["_id"]),
                "title": doc["title"],
                "category": doc["category"]
            } for doc in retrieved_docs]
        )
        
        return QueryResponse(
            answer=llm_response,
            source_documents=source_documents
        )
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@chatbot_router.get("/document/{document_id}", tags=["chatbot"])
async def get_document_content(
    document_id: str,
    current_user: dict = Depends(get_current_user)
):
    try:
        # Get document collection
        documents = get_document_collection()
        
        # Fetch document
        doc = await documents.find_one({"_id": ObjectId(document_id)})
        
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
            
        # Check if user has access
        if current_user["role"] not in doc.get("allowed_roles", []):
            raise HTTPException(status_code=403, detail="You don't have permission to access this document")
            
        # Return document content
        return {
            "id": str(doc["_id"]),
            "title": doc["title"],
            "category": doc["category"],
            "content": doc["document"],
            "description": doc.get("description", "")
        }
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error retrieving document: {str(e)}")

@chatbot_router.get("/history", tags=["chatbot"])
async def get_history(
    limit: int = 10,
    current_user: dict = Depends(get_current_user)
):
    """Get chat history for the current user"""
    try:
        username = current_user["username"]
        history = await get_user_chat_history(username, limit=limit)
        return history
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving chat history: {str(e)}")

@chatbot_router.get("/models", response_model=List[ModelInfo], tags=["chatbot"])
async def list_available_models(current_user: dict = Depends(get_current_user)):
    """List available models that can be used with the chatbot"""
    return LangChainRAG.get_available_models()

@chatbot_router.post("/test-model", tags=["chatbot"])
async def test_model(
    request: TestModelRequest,
    current_user: dict = Depends(get_current_user)
):
    """Test LangChain model with a direct prompt (admin only)"""
    # Only allow c-level-executive to test models directly
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can test models directly")
        
    try:
        # Initialize LangChain with the requested model
        rag = LangChainRAG(model_name=request.model)
        
        # Use direct chat completion for testing
        response = await rag.direct_chat_completion(request.prompt)
        
        return {
            "model": request.model,
            "response": response
        }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error testing model: {str(e)}") 