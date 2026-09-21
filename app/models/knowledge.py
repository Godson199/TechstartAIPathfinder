"""
KnowledgeChunk model — one row per chunk of the Participant Handbook,
curriculum, or FAQ, with an OpenAI embedding stored as JSON text.

This avoids requiring the PostgreSQL vector extension on local development servers.
"""
from sqlalchemy import Column, Integer, String, Text

from app.core.database import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)   # "handbook" | "curriculum" | "faq"
    section = Column(String, nullable=False)  # e.g. "Week 6 — SQL Fundamentals"
    content = Column(Text, nullable=False)
    embedding = Column(Text, nullable=False)
