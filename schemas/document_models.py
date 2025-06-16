from pydantic import BaseModel
from typing import List, Optional

class DocumentAccess(BaseModel):
    """Define which roles have access to a document"""
    document_id: str
    title: str
    allowed_roles: List[str]
    description: Optional[str] = None

class RolePermission(BaseModel):
    """Define permissions for each role"""
    role: str
    description: str
    can_view_finance: bool = False
    can_view_marketing: bool = False
    can_view_hr: bool = False
    can_view_engineering: bool = False
    can_view_company_general: bool = True  # All roles can view general info

# Predefined role permissions
DEFAULT_ROLE_PERMISSIONS = {
    "finance": RolePermission(
        role="finance",
        description="Finance team members",
        can_view_finance=True,
        can_view_marketing=False,
        can_view_hr=False,
        can_view_engineering=False
    ),
    "marketing": RolePermission(
        role="marketing",
        description="Marketing team members",
        can_view_finance=False,
        can_view_marketing=True,
        can_view_hr=False,
        can_view_engineering=False
    ),
    "hr": RolePermission(
        role="hr",
        description="HR team members",
        can_view_finance=False,
        can_view_marketing=False,
        can_view_hr=True,
        can_view_engineering=False
    ),
    "engineering": RolePermission(
        role="engineering",
        description="Engineering team members",
        can_view_finance=False,
        can_view_marketing=False,
        can_view_hr=False,
        can_view_engineering=True
    ),
    "c-level-executive": RolePermission(
        role="c-level-executive",
        description="C-level executives with full access",
        can_view_finance=True,
        can_view_marketing=True,
        can_view_hr=True,
        can_view_engineering=True
    ),
    "employee": RolePermission(
        role="employee",
        description="Regular employees with limited access",
        can_view_finance=False,
        can_view_marketing=False,
        can_view_hr=False,
        can_view_engineering=False
    )
}

class QueryRequest(BaseModel):
    """Request format for chatbot queries"""
    query: str

class DocumentMetadata(BaseModel):
    """Metadata about a document returned in responses"""
    id: str
    title: str
    category: str

class QueryResponse(BaseModel):
    """Response format for chatbot queries"""
    answer: str
    source_documents: List[DocumentMetadata] 