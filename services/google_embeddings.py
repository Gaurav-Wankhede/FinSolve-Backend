import httpx
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API configuration
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_URL = os.getenv(
    "GOOGLE_EMBEDDING_URL",
    "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent"
)

# Get embedding from Google
async def get_google_embedding(text: str):
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    
    # Pass API key as a parameter instead of in the URL
    params = {"key": GOOGLE_API_KEY}
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            GOOGLE_EMBEDDING_URL,
            json=payload,
            params=params
        )
        data = response.json()
        return data["embedding"]["values"]