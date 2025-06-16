from fastapi import FastAPI
from services.upload_files import upload_files
from services.user.user_role import admin_router
from services.user.user_login import user_login
from services.chatbot import chatbot_router
from services.debug import debug_router
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="FinSolve RAG-RBAC API",
        version="1.0.0",
        description="""
        Role-based access control (RBAC) API with LangChain RAG implementation for FinSolve Technologies.
        
        This API provides:
        - User authentication with JWT tokens
        - Role-based access control
        - Document management with role-specific access
        - LangChain RAG-based chatbot with role-based information retrieval
        - OpenRouter integration for accessing various LLMs
        
        Available roles:
        - finance: Access to financial data
        - marketing: Access to marketing data
        - hr: Access to HR data
        - engineering: Access to engineering data
        - c-level-executive: Full access to all data
        - employee: Access to general company information
        """,
        routes=app.routes,
    )
    
    # Add JWT bearer security scheme
    openapi_schema["components"] = openapi_schema.get("components", {})
    openapi_schema["components"]["securitySchemes"] = {
        "Bearer": {
            "type": "http",
            "scheme": "bearer",
            "bearerFormat": "JWT",
            "description": "Enter JWT token in the format: Bearer your_token"
        }
    }
    
    # Remove global security - we'll apply it per endpoint
    # openapi_schema["security"] = [{"Bearer": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

app.include_router(user_login)
app.include_router(upload_files)
app.include_router(admin_router)
app.include_router(chatbot_router, prefix="/api/v1")
app.include_router(debug_router)