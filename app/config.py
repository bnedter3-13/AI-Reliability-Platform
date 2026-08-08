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
    DATABASE_URL: str = os.environ.get("DATABASE_URL", "sqlite:///./ai_reliability.db")
    DRIFT_ALERT_THRESHOLD: float = float(os.environ.get("DRIFT_ALERT_THRESHOLD", "0.15"))
    MODEL_NAME: str = "claude-sonnet-5"


settings = Settings()
