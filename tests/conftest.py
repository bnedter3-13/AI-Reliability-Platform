"""
tests/conftest.py

Shared fixtures for tests that need a database (MLOps version tracking, AI Analysis
Agent). Uses an in-memory SQLite database so these tests stay fast and don't touch
the real DATABASE_URL - each test gets a fresh, isolated schema.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database.models import Base


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()
