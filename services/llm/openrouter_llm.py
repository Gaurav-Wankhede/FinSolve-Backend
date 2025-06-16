import os
import httpx
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1/chat/completions")

# Default headers for OpenRouter
DEFAULT_HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "HTTP-Referer": "https://finsolve-rag.com",  # Replace with your actual domain
    "X-Title": "FinSolve RAG-RBAC",
    "Content-Type": "application/json"
}

class Message:
    """Simple class to represent a chat message"""
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content
        
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
async def generate_response(
    messages: List[Message],
    model: str = "google/gemini-2.0-flash-exp:free",
    temperature: float = 0.7,
    max_tokens: int = 1000
) -> Dict[str, Any]:
    """
    Generate a response using OpenRouter API.
    
    Args:
        messages: List of Message objects representing the conversation
        model: Model identifier to use
        temperature: Controls randomness (0-1)
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Response from the API
    """
    try:
        payload = {
            "model": model,
            "messages": [msg.to_dict() for msg in messages],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=DEFAULT_HEADERS,
                timeout=60.0
            )
            
            response.raise_for_status()
            return response.json()
            
    except httpx.HTTPStatusError as e:
        print(f"HTTP error: {e}")
        raise
    except httpx.RequestError as e:
        print(f"Request error: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

async def create_rag_prompt(
    query: str, 
    context_docs: List[Dict[str, Any]],
    role: str
) -> List[Message]:
    """
    Create a RAG prompt with system instructions, retrieved context, and user query.
    
    Args:
        query: The user's question
        context_docs: List of retrieved documents to use as context
        role: The user's role for personalized responses
        
    Returns:
        List of Message objects for the LLM prompt
    """
    # Format context from retrieved documents
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        doc_title = doc.get("title", "Untitled")
        doc_category = doc.get("category", "Uncategorized")
        doc_content = doc.get("document", "")
        
        # Add document info and content to context
        context_text += f"\n\nDOCUMENT {i}: {doc_title} (Category: {doc_category})\n{doc_content}"
    
    # Create system message with instructions
    system_prompt = f"""You are an AI assistant for FinSolve Technologies, providing role-specific information to users. 
You're responding to a user with the role: {role}.

Follow these guidelines:
1. Answer questions based ONLY on the context provided below
2. If the information isn't in the context, say "I don't have that information" - DO NOT make up answers
3. Keep responses professional, clear, and concise
4. Include citations to the source documents when appropriate using [Document Title]
5. Focus on providing factual information relevant to the user's role

CONTEXT INFORMATION:
{context_text}
"""

    # Create the conversation
    messages = [
        Message(role="system", content=system_prompt),
        Message(role="user", content=query)
    ]
    
    return messages

async def generate_rag_response(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    user_role: str,
    model: str = "google/gemini-2.0-flash-exp:free"
) -> str:
    """
    Generate a response using RAG with OpenRouter.
    
    Args:
        query: User's question
        retrieved_docs: Documents retrieved from vector search
        user_role: User's role in the system
        model: LLM model to use
        
    Returns:
        Generated response text
    """
    # Create RAG prompt
    messages = await create_rag_prompt(query, retrieved_docs, user_role)
    
    # Generate response using OpenRouter
    response = await generate_response(
        messages=messages,
        model=model,
        temperature=0.7,
        max_tokens=1000
    )
    
    # Extract text from response
    try:
        return response["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        print(f"Error extracting response content: {e}")
        return "I apologize, but I encountered an error generating a response." 