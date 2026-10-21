# Deploy TechieStart Chatbot to Vercel with Neon

## Local setup

Install the project dependencies, link the checkout to the Vercel project, and
pull its environment variables:

```powershell
pip install -r requirements.txt
vercel link
vercel env pull .env.local
```

`.env.local` is gitignored. The database module reads it only for local runs.
The schema and ingestion scripts use `DATABASE_URL_UNPOOLED` when available,
because one-off scripts should connect directly rather than use Neon’s
serverless pooler.

## Prepare Neon

In the Neon SQL Editor, enable pgvector once:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

Create all application tables and ingest the checked-in FAQ/curriculum content
from the repository root:

```powershell
python scripts/create_tables.py
python scripts/ingest_knowledge.py
```

Both scripts prefer `DATABASE_URL_UNPOOLED` if Vercel supplied it in
`.env.local`. `knowledge_chunks.embedding` is `vector(1536)`, matching the
default `text-embedding-3-small` model. If you change `EMBEDDING_MODEL`, it
must still return 1,536 dimensions unless you also change the model column and
recreate/migrate the table.

Verify ingestion in Neon SQL Editor:

```sql
SELECT source, COUNT(*) AS rows
FROM knowledge_chunks
GROUP BY source
ORDER BY source;
```

## Vercel environment variables

The Neon Storage integration supplies `DATABASE_URL` for the deployed app and
normally `DATABASE_URL_UNPOOLED` for scripts.

| Variable | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | Yes | Gemini chat and Pathfinder intent/explanations |
| `API_KEY` or `OPENAI_API_KEY` | Yes for ingestion/semantic retrieval | OpenAI-compatible embeddings |
| `OPENAI_BASE_URL` | Optional | Defaults to `https://openrouter.ai/api/v1` |
| `EMBEDDING_MODEL` | Optional | Defaults to `text-embedding-3-small` (1,536 dimensions) |
| `PATHFINDER_PROGRAM_CATALOG` | Optional | Defaults to the committed `data/programs.json` |
| `DATABASE_URL` | From Neon | Pooled Neon URL used by the running app |
| `DATABASE_URL_UNPOOLED` | From Neon | Direct URL used by local maintenance scripts |

Keep `data/faq.md` and `data/programs.json` committed. The app reads them using
paths based on `__file__`, not by writing to the Vercel filesystem.

## Deploy and smoke test

The project’s `vercel.json` builds `main.py`; that module exports the FastAPI
`app`, includes `bot_router`, and provides `GET /`. Schema creation is an
explicit local task, not a serverless startup operation.

Deploy:

```powershell
vercel --prod
```

Replace `<production-url>` with the deployment URL printed by Vercel:

```powershell
curl.exe https://<production-url>/
curl.exe -X POST https://<production-url>/api/bot/chat `
  -H "Content-Type: application/json" `
  -d '{"message":"What programs do you offer?","session_id":"test-1"}'
curl.exe -X POST https://<production-url>/api/bot/pathfinder/start `
  -H "Content-Type: application/json" `
  -d '{"session_id":"test-1"}'
```

Expected responses:

- Health: `{"status":"ok","service":"TechieStart Chatbot"}`. This checks the
  app route, not database connectivity.
- Chat: HTTP 200 with `{"reply":"..."}`. With knowledge ingested, the reply to
  a TechieStart information question should be grounded in the FAQ/curriculum.
- Pathfinder start: HTTP 200 with `{"reply":"What area of technology interests
  you most right now?"}` (the reply is JSON-escaped on one line).

Common failures:

- **SSL/connection errors:** confirm the Neon URL includes `sslmode=require`
  and that the correct Neon project/database URL was pulled. The URL scheme
  can be `postgres://` or `postgresql://`; the app rewrites it for psycopg 3.
- **Empty or generic knowledge answers:** check the Neon row counts above,
  then confirm ingestion used the intended `DATABASE_URL_UNPOOLED` and that
  `data/faq.md` / curriculum files contain the expected content.
- **Missing API-key errors:** add `GEMINI_API_KEY` for chat/Pathfinder and
  `API_KEY` or `OPENAI_API_KEY` for embeddings/ingestion in the Vercel project.
- **Cold-start timeout:** retry once; initial serverless startup and external
  Gemini/embedding/database connections may take longer than warm requests.
