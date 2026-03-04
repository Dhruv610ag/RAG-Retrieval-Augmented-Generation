import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError("GROQ_API_KEY not found in .env file")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise EnvironmentError("GOOGLE_API_KEY not found in .env file")

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
if not TAVILY_API_KEY:
    raise EnvironmentError("TAVILY_API_KEY not found in .env file")

GROQ_MODEL="meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_TEMPERATURE=0.7

GOOGLE_MODEL="gemini-3.0-pro"
GOOGLE_TEMPERATURE=0.7

EMBEDDING_MODEL="embeddinggemma:latest"

CHUNK_SIZE=600
CHUNK_OVERLAP=150

SEARCH_VECTOR={"k":4}