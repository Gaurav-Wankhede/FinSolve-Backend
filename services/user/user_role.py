from fastapi import APIRouter, HTTPException, Depends
from utils.mongo import get_role_collection
from services.chatbot import get_current_user
from schemas.document_models import RolePermission as RoleUpdate

admin_router = APIRouter()

@admin_router.post("/roles", tags=["roles"])
async def add_role(role: RoleUpdate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can add roles.")
    roles = get_role_collection()
    await roles.insert_one(role.dict())
    return {"message": "Role added."}

@admin_router.put("/roles/{role_name}", tags=["roles"])
async def update_role(role_name: str, role: RoleUpdate, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can update roles.")
    roles = get_role_collection()
    await roles.update_one({"role": role_name}, {"$set": role.dict()})
    return {"message": "Role updated."}

@admin_router.delete("/roles/{role_name}", tags=["roles"])
async def delete_role(role_name: str, current_user: dict = Depends(get_current_user)):
    if current_user["role"] != "c-level-executive":
        raise HTTPException(status_code=403, detail="Only C-Level Executives can delete roles.")
    roles = get_role_collection()
    await roles.delete_one({"role": role_name})
    return {"message": "Role deleted."}