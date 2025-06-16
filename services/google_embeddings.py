import httpx
from utils.google import GOOGLE_EMBEDDING_URL

# Get embedding from Google
async def get_google_embedding(text: str):
    payload = {
        "model": "models/embedding-001",
        "content": {"parts": [{"text": text}]}
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(GOOGLE_EMBEDDING_URL, json=payload)
        data = response.json()
        return data["embedding"]["values"]

