from fastapi import HTTPException, Depends, Security
from utils.mongo import get_user_collection
from jose import jwt, JWTError
from fastapi.security import HTTPAuthorizationCredentials
from bson import ObjectId
from schemas.user_db import LoginRequest, UserCreate
from utils.config import admin_router, user_login, security
from utils.config import SECRET_KEY, ALGORITHM

# Helper function to convert MongoDB documents to JSON-serializable dicts
def convert_mongo_doc(doc):
    if doc is None:
        return None
    
    if isinstance(doc, list):
        return [convert_mongo_doc(item) for item in doc]
    
    if isinstance(doc, dict):
        result = {}
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                result[key] = str(value)
            elif isinstance(value, (dict, list)):
                result[key] = convert_mongo_doc(value)
            else:
                result[key] = value
        return result
    
    return doc

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

@user_login.post("/login", tags=["authentication"])
async def login(data: LoginRequest):
    users = get_user_collection()
    user = await users.find_one({"username": data.username})
    if not user or user["password"] != data.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    # No 'exp' in payload = token never expires
    token = jwt.encode({"username": user["username"], "role": user["role"]}, SECRET_KEY, algorithm=ALGORITHM)
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

@user_login.get("/user", tags=["users"])
async def get_users(current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can get users.")
    users = get_user_collection()
    result = []
    async for user in users.find({}):
        result.append(convert_mongo_doc(user))
    return result

@user_login.post("/users", tags=["users"])
async def add_user(user: UserCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can add users.")
    users = get_user_collection()
    # Check if user already exists
    existing = await users.find_one({"username": user.username})
    if existing:
        raise HTTPException(status_code=400, detail="User already exists.")
    await users.insert_one(user.dict())
    return {"message": "User added."}

@user_login.put("/users/{user_id}", tags=["users"])
async def update_user(user_id: str, user: UserCreate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can update users.")
    users = get_user_collection()
    await users.update_one({"_id": ObjectId(user_id)}, {"$set": user.dict()})
    return {"message": "User updated."}

