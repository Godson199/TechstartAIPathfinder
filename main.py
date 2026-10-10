import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.bot.router import router as bot_router
from app.core.database import Base, engine

load_dotenv(Path(__file__).resolve().parent / ".env")

os.environ.setdefault("API_PREFIX", "/api")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        Base.metadata.create_all(bind=engine)
    except Exception as exc:
        print("Startup database initialization failed:", type(exc).__name__, str(exc))
    yield


app = FastAPI(title="TechieStart Chatbot RAG - Minimal API", lifespan=lifespan)

# --- CORS: allow the website to call this API from the browser ---
_default_origins = "https://techiestart.com,https://www.techiestart.com,http://localhost:5173"
ALLOWED_ORIGINS = [
    o.strip()
    for o in os.getenv("ALLOWED_ORIGINS", _default_origins).split(",")
    if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    allow_credentials=False,
)

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