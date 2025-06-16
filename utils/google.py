import os
from dotenv import load_dotenv

load_dotenv()

# Google AI Studio API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_EMBEDDING_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/embedding-001:embedContent?key="
    + GOOGLE_API_KEY
)