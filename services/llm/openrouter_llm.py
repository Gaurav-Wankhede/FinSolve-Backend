import os
import httpx
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()

# Get API credentials - with fallback warning for missing key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    import logging
    logger = logging.getLogger(__name__)
    logger.error("OPENROUTER_API_KEY is not set in .env file. API calls will fail with 401 Unauthorized.")

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
        
        # Debug the API key (first 4 chars and last 4 chars only)
        api_key = OPENROUTER_API_KEY
        if api_key:
            print(f"Using API key: {api_key[:4]}...{api_key[-4:]}")
        else:
            print("API key is not set!")
        
        # Explicitly set headers for this request
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://finsolve-rag.com",
            "X-Title": "FinSolve RAG-RBAC",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OPENROUTER_API_URL,
                json=payload,
                headers=headers,  # Use the explicitly defined headers
                timeout=60.0
            )
            
            print(f"OpenRouter response status: {response.status_code}")
            if response.status_code != 200:
                print(f"Error response: {response.text}")
            
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
        doc_format = doc.get("format", "text").lower()
        
        # Add document info and content to context with format-specific instructions
        context_text += f"\n\nDOCUMENT {i}: {doc_title} (Category: {doc_category}, Format: {doc_format})\n"
        
        # Add format-specific handling instructions
        if doc_format == "csv":
            context_text += "The following contains CSV data. Parse it as a table structure with columns and rows:\n"
            
            # Add header information if available
            if "csv_headers" in doc:
                headers = doc.get("csv_headers", [])
                context_text += f"CSV Headers: {', '.join(headers)}\n"
            
            # Special handling for HR data
            if doc.get("hr_data", False):
                context_text += "This is HR-specific data. When responding to queries about employees, format the data in a structured way and ensure proper interpretation of employee records, departments, and personnel data.\n"
        
        elif doc_format == "markdown":
            context_text += "The following contains Markdown-formatted data. Interpret Markdown syntax correctly:\n"
        
        context_text += doc_content

    # Create system message with enhanced instructions for CSV handling
    system_prompt = f"""You are an AI assistant for FinSolve Technologies, providing role-specific information to users. 
You're responding to a user with the role: {role}.

Follow these guidelines:
1. Answer questions based ONLY on the context provided below
2. If the information isn't in the context, say "I don't have that information" - DO NOT make up answers
3. Keep responses professional, clear, and concise
4. Include citations to the source documents when appropriate using [Document Title]
5. Focus on providing factual information relevant to the user's role

Note that all documents come from a MongoDB vector database where they were stored as embeddings. 
The original structure may need to be reconstructed from the text.

For CSV data:
- The data was originally in CSV format before being stored in the vector database
- Reconstruct the table structure from the comma-separated values
- Treat the first line as headers unless clearly not appropriate
- Parse CSV as structured tables with rows and columns
- For HR data, organize employee information clearly by columns
- Present tabular data in a readable format
- When dealing with employee records, include relevant fields like name, position, department, etc.
- Format numbers and dates according to their meaning (currencies, percentages, dates)

For Markdown data:
- Properly interpret headers, lists, tables, and other Markdown elements
- Maintain the structure and formatting implied by the Markdown

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