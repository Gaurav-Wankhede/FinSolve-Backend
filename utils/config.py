import os
from fastapi import APIRouter
from fastapi.security import HTTPBearer

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")

admin_router = APIRouter()
user_login = APIRouter()

security = HTTPBearer(
    scheme_name="Bearer",
    description="JWT token authentication",
    auto_error=True
)