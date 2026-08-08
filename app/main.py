"""
app/main.py

FastAPI application entrypoint for the AI Reliability Platform (AI Doctor).
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.database.connection import init_db
from app.api.routes import router as api_router

DASHBOARD_PATH = Path(__file__).parent / "dashboard" / "index.html"

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Reliability Platform",
    version="0.1.0",
    description="Monitors, evaluates, and improves the reliability of AI/RAG systems.",
)

app.include_router(api_router)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def root() -> dict[str, str]:
    return {"message": "AI Reliability Platform is running"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy", "service": "ai-reliability-platform"}


@app.get("/dashboard")
def dashboard() -> FileResponse:
    return FileResponse(DASHBOARD_PATH)
