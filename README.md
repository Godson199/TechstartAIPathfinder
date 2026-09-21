# TechieStart Chatbot — RAG Module

Grounded chatbot module: answers only from your FAQ and program information
(no improvising on fees, policies, or program details).

## Files

```
app/
  core/database.py     # DB setup — merge into your existing one, don't overwrite
  models/chat.py        # conversation history table
  models/knowledge.py   # embedded knowledge chunks table (pgvector)
  models/pathfinder.py   # AI Pathfinder session state table
  bot/router.py          # /api/bot/chat + /api/bot/pathfinder/start (RAG + Gemini)
  bot/pathfinder.py      # Pathfinder questions, session state, matching engine
scripts/
  ingest_knowledge.py    # chunks + embeds your docs into Postgres
data/
  faq.md                     # template — replace with your real FAQ
  programs.example.json      # schema template for the Pathfinder's program catalog — copy to programs.json and fill in real data
requirements.txt
tests/                   # unit + integration tests — run with `pytest`
```

## Setup steps

1. **Merge files into your existing FastAPI backend repo**, replacing the
   current `app/bot/` FAQ logic. Keep your existing `app/core/database.py`
   if you already have one — just make sure it exposes `Base`,
   `SessionLocal`, and `get_db` as shown here.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Enable pgvector on your Render Postgres** (run once, via psql or a
   migration):
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

4. **Create the tables** (Alembic migration, or quick-and-dirty for now):
   ```python
   from app.core.database import Base, engine
   from app.models import chat, knowledge
   Base.metadata.create_all(bind=engine)
   ```

5. **Add your real content** to `data/`:
   ```bash
   pandoc participant_handbook.docx -o data/participant_handbook.md
   pandoc curriculum.docx -o data/curriculum.md
   # edit data/faq.md directly
   ```
   You can also place the supplied FAQ CSV at
   `data/TechieStart_50_FAQs_Guide.csv`; rows with `Question`, `Answer`, and
   optional `Category` columns are imported automatically. See `data/README.md`
   for the complete document drop zone.

6. **Set environment variables** (Render dashboard):
   - `GEMINI_API_KEY` — chat model (conversation + Pathfinder explanations)
   - Gemini chat uses stable `gemini-flash-lite-latest`
   - `API_KEY` or `OPENAI_API_KEY` — embeddings only, via OpenRouter/OpenAI
   - `OPENAI_BASE_URL` (optional, defaults to OpenRouter)
   - `EMBEDDING_MODEL` (optional, defaults to `text-embedding-3-small`)
   - `PATHFINDER_PROGRAM_CATALOG` (optional, defaults to `data/programs.json`)
   - `DATABASE_URL` (you already have this)

7. **Run ingestion** (locally against your Render Postgres, or as a one-off
   Render job):
   ```bash
   python scripts/ingest_knowledge.py
   ```

8. **Mount the router** in your main FastAPI app:
   ```python
   from app.bot.router import router as bot_router
   app.include_router(bot_router)
   ```

9. **Redeploy.** Your existing frontend widget doesn't need changes — it
   already POSTs `{ message, session_id }` to `/api/bot/chat` and expects
   `{ reply }` back.

## AI Pathfinder

The Pathfinder extends the existing chat endpoint without creating a second
chat system:

```text
POST /api/bot/pathfinder/start
{ "session_id": "browser-session-id" }
```

Participants can also start it conversationally through `/api/bot/chat` with
messages such as `Which course is best for me?`, or restart at any point with
phrases like `restart` or `explore another program` — this works whether a
session is mid-assessment, just completed, or was abandoned. Their answers
are stored in the `pathfinder_sessions` table. Normal TechieStart questions
(fees, registration, etc.) are detected mid-assessment and answered through
the existing RAG flow without losing the participant's place in the
Pathfinder — they pick up on the same question afterwards.

**Recommendations are intentionally disabled until authoritative program data
is provided.** Copy `data/programs.example.json` to `data/programs.json` (or
point `PATHFINDER_PROGRAM_CATALOG` elsewhere) and fill in the actual program
IDs, names, descriptions, suitable-for profiles, prerequisites, skills,
learning outcomes, career directions, and durations — none of this exists in
the current FAQ knowledge base, so it has to come from TechieStart's program
team. Until that file exists, the Pathfinder still runs the full
conversation but tells the participant honestly that no catalog is
configured yet, rather than guessing.

Once a catalog is in place, the matching engine (`match_programs` in
`app/bot/pathfinder.py`) scores every program with transparent, rule-based
keyword overlap — no LLM involved in deciding the match, so it's explainable
and unit-testable on its own. Gemini is only used afterwards, in
`explain_recommendation` (`app/bot/router.py`), to turn the winning match
into a warm explanation — strictly grounded in the match factors and in RAG
context retrieved for that specific program, with instructions not to invent
details. If that generation step fails for any reason, the deterministic
match summary is shown instead, so a recommendation is never lost to an API
hiccup.

## Re-ingesting after content updates

Just re-run `python scripts/ingest_knowledge.py` — it clears and re-embeds
each source (handbook/curriculum/faq) it's given, so it's safe to run
repeatedly whenever you update the docs.
