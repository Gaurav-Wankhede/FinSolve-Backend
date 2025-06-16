from fastapi import APIRouter, File, Form, UploadFile, HTTPException, Depends, Header, Security
from fastapi.responses import JSONResponse
from utils.mongo import get_document_collection
from jose import jwt, JWTError
import os
from services.google_embeddings import get_google_embedding
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId
from typing import Optional
from schemas.document_models import DocumentAccess

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

upload_files = APIRouter()

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

# Endpoint for upload
@upload_files.post("/upload", tags=["documents"])
async def upload_file(
    title: str = Form(...),
    file: UploadFile = File(...),
    category: str = Form(...),  # finance, marketing, hr, engineering, general
    description: Optional[str] = Form(None),
    allowed_roles: Optional[str] = Form("c-level-executive"),  # comma-separated roles
    current_user: dict = Depends(get_current_user)
):
    # Only allow c-level-executive
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can upload files.")
    
    # Validate category
    valid_categories = ["finance", "marketing", "hr", "engineering", "general"]
    if category not in valid_categories:
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"}
        )
    
    try:
        contents = await file.read()
        content_str = contents.decode("utf-8")

        # Check allowed file types
        if not (file.filename.endswith(".md") or file.filename.endswith(".csv") or file.filename.endswith(".txt")):
            return JSONResponse(
                status_code=400,
                content={"error": "Only .md, .csv, and .txt files are supported."}
            )

        # Chunking logic (simple line-based chunks)
        chunks = content_str.split("\n\n")  # Or use smarter recursive chunking

        # Get embeddings for all chunks
        embeddings = []
        for chunk in chunks:
            if chunk.strip():  # skip empty chunks
                emb = await get_google_embedding(chunk)
                embeddings.append(emb)

        # Parse allowed roles
        role_list = [role.strip() for role in allowed_roles.split(",")]
        
        # Add category-specific roles automatically
        if category == "finance" and "finance" not in role_list:
            role_list.append("finance")
        elif category == "marketing" and "marketing" not in role_list:
            role_list.append("marketing")
        elif category == "hr" and "hr" not in role_list:
            role_list.append("hr")
        elif category == "engineering" and "engineering" not in role_list:
            role_list.append("engineering")
        
        # C-level executives always have access
        if "c-level-executive" not in role_list:
            role_list.append("c-level-executive")
            
        # If general category, all roles have access
        if category == "general":
            role_list = ["finance", "marketing", "hr", "engineering", "c-level-executive", "employee"]
                
        # Save to MongoDB
        doc = {
            "title": title,
            "uploader": current_user["username"],
            "document": content_str,
            "embedding_chunks": embeddings,
            "category": category,
            "description": description,
            "allowed_roles": role_list
        }
        result = await get_document_collection().insert_one(doc)

        return JSONResponse(
            status_code=200,
            content={"message": "File uploaded and embedded.", "id": str(result.inserted_id)}
        )
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@upload_files.put("/upload/{document_id}", tags=["documents"])
async def update_file(
    document_id: str,
    title: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    category: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    allowed_roles: Optional[str] = Form(None),
    current_user: dict = Depends(get_current_user)
):
    # Only allow c-level-executive
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can update files.")
    
    try:
        # Find existing document
        document_collection = get_document_collection()
        existing_doc = await document_collection.find_one({"_id": ObjectId(document_id)})
        if not existing_doc:
            return JSONResponse(status_code=404, content={"error": f"No document with id '{document_id}' found."})

        # Prepare update data
        update_data = {}
        
        if title:
            update_data["title"] = title
            
        if category:
            valid_categories = ["finance", "marketing", "hr", "engineering", "general"]
            if category not in valid_categories:
                return JSONResponse(
                    status_code=400,
                    content={"error": f"Invalid category. Must be one of: {', '.join(valid_categories)}"}
                )
            update_data["category"] = category
            
        if description:
            update_data["description"] = description
            
        if allowed_roles:
            role_list = [role.strip() for role in allowed_roles.split(",")]
            # Ensure c-level-executive is always included
            if "c-level-executive" not in role_list:
                role_list.append("c-level-executive")
            update_data["allowed_roles"] = role_list
            
        if file:
            contents = await file.read()
            content_str = contents.decode("utf-8")

            if not (file.filename.endswith(".md") or file.filename.endswith(".csv") or file.filename.endswith(".txt")):
                return JSONResponse(status_code=400, content={"error": "Only .md, .csv, and .txt files are supported."})

            # Chunking and embedding
            chunks = content_str.split("\n\n")
            embeddings = []
            for chunk in chunks:
                if chunk.strip():
                    emb = await get_google_embedding(chunk)
                    embeddings.append(emb)
                    
            update_data["document"] = content_str
            update_data["embedding_chunks"] = embeddings

        if not update_data:
            return JSONResponse(status_code=400, content={"error": "No update data provided"})

        # Update document
        result = await document_collection.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": update_data}
        )

        return JSONResponse(
            status_code=200,
            content={"message": f"Document updated successfully."}
        )

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Get all documents with role-based filtering
@upload_files.get("/documents", tags=["documents"])
async def get_documents(current_user: dict = Depends(get_current_user)):
    try:
        documents = get_document_collection()
        
        # Build query based on user role
        query = {"allowed_roles": current_user["role"]}
        
        # Project only needed fields
        projection = {
            "title": 1,
            "category": 1,
            "description": 1,
            "uploader": 1,
            "allowed_roles": 1
        }
        
        result = []
        async for doc in documents.find(query, projection):
            # Convert ObjectId to string
            doc["_id"] = str(doc["_id"])
            result.append(doc)
            
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")
