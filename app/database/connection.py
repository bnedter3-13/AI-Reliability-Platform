"""
app/database/connection.py

SQLAlchemy engine and session setup. Defaults to a local SQLite file so the project
runs with zero external setup; point DATABASE_URL at PostgreSQL for production.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from app.config import settings

connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(settings.DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Session:
    """FastAPI dependency — yields a DB session and ensures it's closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create all tables. Call once at startup (see main.py)."""
    from app.database import models  # noqa: F401 (ensures models are registered)
    from app.database.models import Base
    Base.metadata.create_all(bind=engine)
