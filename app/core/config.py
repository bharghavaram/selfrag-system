import os
from dotenv import load_dotenv
load_dotenv()

class Settings:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o")
    EMBED_MODEL: str = os.getenv("EMBED_MODEL", "text-embedding-3-small")
    FAISS_INDEX_PATH: str = os.getenv("FAISS_INDEX_PATH", "data/faiss_index")
    MAX_RETRIEVE: int = int(os.getenv("MAX_RETRIEVE", "5"))
    MAX_REFLECTION_ROUNDS: int = int(os.getenv("MAX_REFLECTION_ROUNDS", "3"))
    RELEVANCE_THRESHOLD: float = float(os.getenv("RELEVANCE_THRESHOLD", "0.7"))
    SUPPORT_THRESHOLD: float = float(os.getenv("SUPPORT_THRESHOLD", "0.7"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "2048"))
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8000"))

settings = Settings()
