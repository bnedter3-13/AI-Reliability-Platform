"""
app/config.py

Central place to load environment variables. Import settings from here instead of
calling os.environ.get() scattered across the codebase.
"""

import os
from dotenv import load_dotenv

load_dotenv()  # reads .env if present; safe no-op in production if env vars are set another way


class Settings:
    ANTHROPIC_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
    OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")
    GEMINI_API_KEY: str | None = os.environ.get("GEMINI_API_KEY")
    QWEN_API_KEY: str | None = os.environ.get("QWEN_API_KEY")  # holds an OpenRouter API key (openrouter.ai)
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./ai_reliability.db")
    DRIFT_ALERT_THRESHOLD: float = float(os.environ.get("DRIFT_ALERT_THRESHOLD", "0.15"))
    MODEL_NAME: str = "claude-sonnet-5"

    # Knowledge Base + Retrieval Verification Agent (Component 11): where the local
    # Chroma vector store persists its data on disk.
    CHROMA_PERSIST_DIR: str = os.environ.get("CHROMA_PERSIST_DIR", "./data/chroma_db")


settings = Settings()
