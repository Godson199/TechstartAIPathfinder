"""
ChatMessage model — stores every user/assistant turn, keyed by session_id.
session_id is generated client-side (e.g. crypto.randomUUID()) and persisted
in localStorage so a returning visitor keeps their conversation history.
"""
from datetime import datetime

from sqlalchemy import Column, Integer, String, Text, DateTime

from app.core.database import Base


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True, nullable=False)
    role = Column(String, nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
