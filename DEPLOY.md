# DEPLOY.md — TechieStart Chatbot (RAG) on Render

Deployment steps for the grounded chatbot module. Assumes the module is
already integrated into your FastAPI backend.

## 1. Enable pgvector on your Render Postgres

Render's managed Postgres supports the `vector` extension, but it must be
enabled per-database.

1. Open the Render dashboard → your Postgres instance → **Connect** → copy
   the **External Connection String** (psql).
2. Connect and run:
   ```bash
   psql "<external-connection-string>"
   ```
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```
3. Confirm it's active:
   ```sql
   SELECT extname FROM pg_extension WHERE extname = 'vector';
   ```

## 2. Merge the code into your backend repo

- Copy `app/bot/router.py`, `app/models/chat.py`, `app/models/knowledge.py`
  into your existing repo structure.
- **Do not overwrite** your existing `app/core/database.py` — just confirm
  it exports `Base`, `SessionLocal`, `get_db` matching this module's
  expectations.
- Merge `requirements.txt` entries (`openai`, `pgvector`, `tiktoken`,
  `pydantic`) into your existing `requirements.txt`.
- Add the router mount in your main app file (e.g. `main.py`):
  ```python
  from app.bot.router import router as bot_router
  app.include_router(bot_router)
  ```

## 3. Set environment variables on Render

Backend service → **Environment**:

| Key | Value |
|---|---|
| `GEMINI_API_KEY` | your Gemini key — used for chat + Pathfinder explanations |
| Gemini model | `gemini-flash-lite-latest` |
| `API_KEY` or `OPENAI_API_KEY` | your OpenAI/OpenRouter key — embeddings only |
| `PATHFINDER_PROGRAM_CATALOG` | path to your program catalog (optional, defaults to `data/programs.json`) |

`DATABASE_URL` should already be set from your existing deployment.

See `data/programs.example.json` for the catalog schema — copy it to
`data/programs.json` and fill in TechieStart's real program data before
participants will get Pathfinder recommendations.

## 4. Create the new tables

Simplest path — add a one-off startup hook or run once via Render Shell:

```bash
# Render dashboard → your backend service → Shell
python -c "
from app.core.database import Base, engine
from app.models import chat, knowledge
Base.metadata.create_all(bind=engine)
"
```

If you're using Alembic migrations for the rest of the schema, generate a
migration instead so it stays consistent with your existing migration
history:

```bash
alembic revision --autogenerate -m "add chat_messages and knowledge_chunks"
alembic upgrade head
```

## 5. Add your real content and ingest it

Locally (against the same `DATABASE_URL` Render uses — pull it from the
Render dashboard into your local `.env` temporarily):

```bash
pandoc participant_handbook.docx -o data/participant_handbook.md
pandoc curriculum.docx -o data/curriculum.md
# edit data/faq.md with your real Q&A content

export DATABASE_URL="<render-external-connection-string>"
export OPENAI_API_KEY="<your-key>"
python scripts/ingest_knowledge.py
```

Alternatively, run it as a one-off Render Job (Render dashboard → **New** →
**Job**, same repo/build, command `python scripts/ingest_knowledge.py`) so
you never need to expose the external DB string locally.

## 6. Deploy

Push to the branch Render auto-deploys from (same as your existing
frontend/backend split — backend on Render, frontend on Vercel). No frontend
changes are needed; the widget already POSTs to `/api/bot/chat`.

## 7. Verify

```bash
curl -X POST https://careerforge-api-i1v3.onrender.com/api/bot/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test-1", "message": "What tracks does CareerForge offer?"}'
```

You should get back a reply grounded in your actual FAQ/handbook content —
if it's vague or generic, double check step 5 (ingestion) actually ran
against the right `DATABASE_URL`.

## Re-deploying after content updates

No redeploy needed for content changes — just re-run
`python scripts/ingest_knowledge.py` (locally or as a Render Job) after
updating the markdown files in `data/`. Only code changes need a redeploy.
