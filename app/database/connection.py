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
    Lightweight auto-migration (works for both SQLite and PostgreSQL):
    Base.metadata.create_all() only creates tables that don't exist yet — it never
    alters an existing table's schema. Since the team's evaluations table may already
    have real history predating a newly-added column (e.g. evaluator_version), this
    adds any missing columns in place via ALTER TABLE so history isn't lost, on
    whichever database DATABASE_URL points at. This is a simple additive-only
    migration — for anything beyond "add a nullable column" (renames, type changes,
    drops), use a real migration tool (e.g. Alembic) instead.
    """
    from sqlalchemy import inspect, text

    inspector = inspect(engine)
    if "evaluations" not in inspector.get_table_names():
        return  # fresh database, create_all() already built the full schema

    existing_columns = {col["name"] for col in inspector.get_columns("evaluations")}
    if "evaluator_version" not in existing_columns:
        with engine.connect() as conn:
            conn.execute(text("ALTER TABLE evaluations ADD COLUMN evaluator_version VARCHAR"))
            conn.commit()
