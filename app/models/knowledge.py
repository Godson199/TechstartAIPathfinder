"""Knowledge chunks and their 1,536-dimensional embeddings."""
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text

from app.core.database import Base


class KnowledgeChunk(Base):
    __tablename__ = "knowledge_chunks"

    id = Column(Integer, primary_key=True)
    source = Column(String, nullable=False)   # "handbook" | "curriculum" | "faq"
    section = Column(String, nullable=False)  # e.g. "Week 6 — SQL Fundamentals"
    content = Column(Text, nullable=False)
    embedding = Column(Vector(1536), nullable=False)
