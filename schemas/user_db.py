from pydantic import BaseModel

class RoleUpdate(BaseModel):
    role: str
    permissions: list

class LoginRequest(BaseModel):
    username: str
    password: str

class UserCreate(BaseModel):
    username: str
    password: str
    role: str