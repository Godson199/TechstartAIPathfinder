"""Create the application's database tables once against the configured database."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
if not os.getenv("VERCEL"):
    load_dotenv(ROOT / ".env.local")

if os.getenv("DATABASE_URL_UNPOOLED"):
    os.environ["DATABASE_URL"] = os.environ["DATABASE_URL_UNPOOLED"]

from app.core.database import Base, engine
from app.models import chat, knowledge, pathfinder


def main() -> None:
    Base.metadata.create_all(bind=engine)
    print("Created database tables.")


if __name__ == "__main__":
    main()
