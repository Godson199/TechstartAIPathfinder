import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from app.bot.router import router as bot_router
from app.core.database import Base, engine
from app.models.chat import ChatMessage
from app.models.knowledge import KnowledgeChunk
from app.models.pathfinder import PathfinderSession

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ.setdefault("API_PREFIX", "/api")

app = FastAPI(title="TechieStart Chatbot RAG - Minimal API")
app.include_router(bot_router)


@app.on_event("startup")
def startup():
    Base.metadata.create_all(bind=engine)


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002)

 