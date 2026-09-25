"""
Database setup.

NOTE: If your existing FastAPI backend already has an app/core/database.py
(you should — this is the CareerForge backend on Render), do NOT overwrite
it with this file. Just make sure it exports `Base`, `SessionLocal`, and a
`get_db` dependency, matching the shape below, and import from your own file
instead of this one.
"""
import os
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:////tmp/techiestart.db",
)

engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
