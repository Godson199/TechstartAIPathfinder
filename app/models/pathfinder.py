from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.core.database import Base


class PathfinderSession(Base):
    __tablename__ = "pathfinder_sessions"

    id = Column(Integer, primary_key=True)
    session_id = Column(String, unique=True, index=True, nullable=False)
    status = Column(String, nullable=False, default="active")
    current_question = Column(Integer, nullable=False, default=0)
    state_json = Column(Text, nullable=False, default="{}")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
