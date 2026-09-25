import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI

from app.bot.router import router as bot_router
from app.core.database import Base, engine

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ.setdefault("API_PREFIX", "/api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="TechieStart Chatbot RAG - Minimal API", lifespan=lifespan)
app.include_router(bot_router)


@app.get("/")
def health_check():
    return {"status": "ok", "service": "TechieStart Chatbot"}


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
    )

 