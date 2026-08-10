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
    _migrate_add_missing_columns()


def _migrate_add_missing_columns() -> None:
    """
    Lightweight auto-migration for SQLite dev databases: Base.metadata.create_all()
    only creates tables that don't exist yet — it never alters an existing table's
    schema. Since the team's local ai_reliability.db already has real evaluation
    history predating the evaluator_version column, this adds any missing columns
    in place via ALTER TABLE so that history isn't lost. No-op for non-SQLite
    databases (e.g. PostgreSQL in production) — use a real migration tool there.
    """
    if not settings.DATABASE_URL.startswith("sqlite"):
        return

    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "evaluations" not in inspector.get_table_names():
        return  # fresh database, create_all() already built the full schema

    existing_columns = {col["name"] for col in inspector.get_columns("evaluations")}
    if "evaluator_version" not in existing_columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE evaluations ADD COLUMN evaluator_version VARCHAR"))
            conn.commit()
