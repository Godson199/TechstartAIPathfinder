# TechieStart Chat API — Frontend Integration Guide

This guide is for frontend developers integrating the TechieStart chatbot and
AI Pathfinder. The API is a JSON-over-HTTPS FastAPI service. Chat and
Pathfinder messages use the same `session_id` and conversation endpoint.

## Base URL

Use the deployed API's origin, without an endpoint path or trailing slash:

```text
https://YOUR-DEPLOYED-DOMAIN
```

For local development, start the API from the repository root:

```powershell
.\.venv\Scripts\python.exe main.py
```

The local base URL is `http://localhost:8000`. The API endpoint paths shown
below already include `/api`.

## Endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Health check |
| `POST` | `/api/bot/chat` | Send a chat message, Pathfinder request, or Pathfinder answer |
| `POST` | `/api/bot/pathfinder/start` | Explicitly start or restart the Pathfinder |

Requests and responses use JSON. There is no authentication or streaming
protocol configured. Do not send API keys or other server credentials from
the browser.

### Health check

```http
GET /
```

Successful response (`200`):

```json
{
  "status": "ok",
  "service": "TechieStart Chatbot"
}
```

This confirms that the app responds; it does not check database or AI provider
connectivity.

### Send a chat message

```http
POST /api/bot/chat
Content-Type: application/json
```

Request:

```json
{
  "session_id": "persistent-client-generated-id",
  "message": "What programs do you offer?"
}
```

Both fields are required strings. A successful response (`200`) is:

```json
{
  "reply": "..."
}
```

The `reply` is plain text. Render it as text, not HTML. The API does not return
a message ID, conversation history, Pathfinder question index, or structured
program recommendation object.

Use this endpoint for all normal chat and Pathfinder conversation. The
backend determines whether a message is a normal question, an assessment
answer, a Pathfinder start/restart request, or a question asked while an
assessment is in progress.

### Start the Pathfinder explicitly

Call this when the user clicks a dedicated “Find my program” or equivalent
button:

```http
POST /api/bot/pathfinder/start
Content-Type: application/json
```

Request:

```json
{
  "session_id": "persistent-client-generated-id"
}
```

Successful response (`200`):

```json
{
  "reply": "I can help you explore the available TechieStart programs. You can answer in your own words, and you can say 'exit' at any time.\n\nWhat area of technology interests you most right now?"
}
```

Starting the Pathfinder resets its assessment answers for that `session_id`.
The endpoint also adds the opening assistant message to that session's chat
history. To start it from chat instead, send a message such as:

```json
{
  "session_id": "persistent-client-generated-id",
  "message": "Which course is best for me?"
}
```

After it starts, post each participant response to `/api/bot/chat` with the
same `session_id`. The API returns the next prompt in `reply`. There is no
separate answer endpoint.

## Session ID and conversation continuity

Generate one opaque, random ID per browser conversation, then reuse it for
every request to both endpoints. Persist it across page reloads (for example,
in `localStorage`) so the backend can associate chat history and active
Pathfinder progress with the same visitor. Do not use a name, email address,
or other personal information as the ID.

Example browser helper:

```ts
const SESSION_STORAGE_KEY = "techiestart-chat-session";

export function getChatSessionId(): string {
  let sessionId = localStorage.getItem(SESSION_STORAGE_KEY);

  if (!sessionId) {
    sessionId = crypto.randomUUID();
    localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  }

  return sessionId;
}
```

If the visitor clears site storage or uses another browser/device, a new ID
will be created and the prior session will not automatically be recovered.
The API does not currently provide account-based session recovery.

## Browser integration example

Configure the deployed origin in the frontend's public environment
configuration. Do not put server-only secrets in frontend environment
variables.

```ts
const API_BASE_URL = "https://YOUR-DEPLOYED-DOMAIN";

type ChatResponse = {
  reply: string;
};

async function postJson<TResponse>(
  path: string,
  body: Record<string, string>,
): Promise<TResponse> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const detail = await response.text();
    throw new Error(
      `Chat API returned HTTP ${response.status}${detail ? `: ${detail}` : ""}`,
    );
  }

  return (await response.json()) as TResponse;
}

export async function sendChatMessage(message: string): Promise<string> {
  const result = await postJson<ChatResponse>("/api/bot/chat", {
    session_id: getChatSessionId(),
    message,
  });
  return result.reply;
}

export async function startPathfinder(): Promise<string> {
  const result = await postJson<ChatResponse>("/api/bot/pathfinder/start", {
    session_id: getChatSessionId(),
  });
  return result.reply;
}
```

The UI should append the user's message and returned `reply` to the visible
conversation, show a pending state while awaiting the response, and restore
the input if the request fails. Replies may contain multiple paragraphs and
newlines; preserve those when displaying them.

## Pathfinder conversation behavior

- Assessment prompts and answers use the normal `/api/bot/chat` endpoint.
- The API decides when to advance. The frontend should display the returned
  `reply` rather than trying to infer the next question.
- Users can ask regular TechieStart questions during the assessment. The
  answer and the pending Pathfinder prompt may both appear in the same
  `reply`; keep the assessment active.
- Users can send `exit` to leave the assessment. They can request a restart
  with phrases such as `restart` or `explore another program`, or the UI can
  call `/api/bot/pathfinder/start` again.
- Keep using the same `session_id` for the entire interaction, including
  after exiting or restarting.
- Recommendations depend on an authoritative program catalog configured by
  the backend. If it is missing or empty, the assessment can still run, but
  the API cannot provide a catalog-backed recommendation.

## CORS and frontend origins

The browser enforces CORS. The API's default allowed origins are:

- `https://techiestart.com`
- `https://www.techiestart.com`
- `http://localhost:5173`

If the frontend is hosted at a different origin, configure the backend's
`ALLOWED_ORIGINS` environment variable as a comma-separated list of complete
origins (scheme, hostname, and optional port), for example:

```text
https://techiestart.com,https://www.techiestart.com,https://preview.example.com
```

Do not include URL paths. The API allows `GET`, `POST`, and `OPTIONS`, and
the `Content-Type` request header. No cookies or credentialed cross-origin
requests are needed.

## Errors and troubleshooting

| Result | Meaning / suggested action |
| --- | --- |
| `200` | Request succeeded; read the JSON `reply`. |
| `422` | Request body is missing a required field or has the wrong type. Send the required string fields shown above. |
| `5xx` | Backend, database, or upstream service failure. Show a retry message and check deployment logs. |
| Browser CORS error | The frontend's exact origin is not in `ALLOWED_ORIGINS`; add it to the backend configuration and redeploy. |
| Network error / timeout | Check the base URL, deployment status, connectivity, and browser console. |

The API may return a user-facing fallback inside a `200` response when an AI
provider is temporarily unavailable. A `200` therefore means the request was
handled, not necessarily that an external AI provider generated the reply.

Do not retry a chat POST automatically without considering duplicate
messages: the server records submitted messages, so a retry can add the same
message twice. If offering retry, let the user trigger it intentionally.

## PowerShell smoke test

Replace the URL with the deployed domain and run:

```powershell
$uri = "https://YOUR-DEPLOYED-DOMAIN/api/bot/chat"
$body = @{
    session_id = "smoke-test-001"
    message = "What programs do you offer?"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri $uri `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

To start the Pathfinder instead:

```powershell
$uri = "https://YOUR-DEPLOYED-DOMAIN/api/bot/pathfinder/start"
$body = @{
    session_id = "smoke-test-001"
} | ConvertTo-Json

Invoke-RestMethod `
    -Uri $uri `
    -Method Post `
    -ContentType "application/json" `
    -Body $body
```

## Backend setup and deployment

The backend expects database and AI-provider configuration in its deployment
environment. These settings and the Neon schema/ingestion steps are
documented in [DEPLOY.md](./DEPLOY.md). Never expose `.env.local` or provider
credentials to the frontend.

Interactive API documentation is available at `/docs` when served by FastAPI.
