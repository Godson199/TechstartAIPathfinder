"""
CareerForge chatbot endpoint — retrieval-augmented (RAG) so answers are
grounded in the actual Participant Handbook, curriculum, and FAQ content
instead of the model improvising.

Mount this router in your main FastAPI app:

    from app.bot.router import router as bot_router
    app.include_router(bot_router)

Env vars required:
    GEMINI_API_KEY   # CHANGED: chat model is now Gemini
    OPENAI_API_KEY   # still used for embeddings only — see note below
    DATABASE_URL   (already set for your existing Postgres)

FIX (2nd pass): the first version of this file used `google.generativeai`,
which is Google's DEPRECATED legacy SDK (support ended Aug 2025) and an
outdated model name. Corrected below to use the current `google-genai`
package and a current model string.
    pip install google-genai   # NOT google-generativeai
"""
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from openai import OpenAI
from google import genai              # FIXED: current SDK, not google.generativeai
from google.genai import types        # FIXED: config objects live here now

from app.core.database import get_db
from app.models.chat import ChatMessage
from app.models.knowledge import KnowledgeChunk
from app.models.pathfinder import PathfinderSession
from app.bot.pathfinder import (
    completed_match,
    is_normal_question,
    process_pathfinder_message,
    start_pathfinder,
    wants_new_pathfinder,
)

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

router = APIRouter()


def get_client() -> OpenAI:
    # CHANGED: this client is now used for EMBEDDINGS ONLY (chat moved to Gemini below).
    # Left in place because your knowledge_chunks embeddings were generated with this
    # provider/model — switching embeddings to Gemini would require re-embedding your
    # whole knowledge base to keep similarity search consistent. Say the word if you
    # want that migration done too.
    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing API key. Set API_KEY or OPENAI_API_KEY in your .env file.")

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )


# FIXED: the new SDK is client-centric rather than model-object-centric.
# One client is reused across requests; system_instruction/config are passed
# per-call to generate_content instead of baked into a model object.
_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    global _gemini_client
    if _gemini_client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing API key. Set GEMINI_API_KEY in your .env file.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


# Use the stable low-demand alias enabled for the configured API key.
CHAT_MODEL = "gemini-flash-lite-latest"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")  # unchanged — still OpenRouter/OpenAI

SYSTEM_PROMPT = """You are the TechieStart Assistant.

Your job is to help users with questions about TechieStart while also
maintaining a friendly, natural conversation.

RULES:

1. FRIENDLY CONVERSATION
You may respond naturally to simple conversational messages that do not
require information from the CONTEXT, such as:
- "Hello"
- "Hi"
- "How are you?"
- "Good morning"
- "Thank you"
- "You're welcome"
- "Bye"
- "Who are you?"
- "Nice to meet you"

Keep these responses short, warm, and friendly.

2. TECHIESTART INFORMATION
For questions about TechieStart's programs, courses, fees, schedules,
registration, requirements, instructors, services, or any other specific
TechieStart information, answer ONLY using the CONTEXT provided below.

The CONTEXT comes from the official TechieStart FAQ and program information.

3. MIXED QUESTIONS
If a message contains both casual conversation and a TechieStart question,
respond naturally to the casual part and answer the TechieStart question using
ONLY the CONTEXT.

For example, if the user says:
"Hi, how are you? Also, when does the AI program start?"

Respond naturally to the greeting, then answer the program question using only
the information available in the CONTEXT.

4. IF INFORMATION IS NOT IN THE CONTEXT
If a user asks a TechieStart-related question and the answer cannot be found
in the CONTEXT, do not guess, assume, or invent information.

Instead, politely say that you are not sure and suggest that the user contact
the TechieStart support line for accurate information.

5. DO NOT INVENT DETAILS
Never make up:
- Program details
- Fees
- Dates
- Schedules
- Course content
- Requirements
- Contact information
- Statistics
- Policies
- Promises or guarantees

6. STYLE
Be concise, friendly, helpful, and encouraging.
Avoid unnecessarily long explanations.

7. WHATSAPP SUPPORT
If the user asks to speak with a human, contact support, continue the
conversation on WhatsApp, or requests assistance that requires a human,
politely offer to transfer them to TechieStart support on WhatsApp.

If the requested TechieStart information is not available in the CONTEXT,
you may also suggest continuing with the TechieStart support team on WhatsApp.

Do not invent or guess the WhatsApp number. Only provide the WhatsApp link
or number if it is explicitly provided in the CONTEXT or application
configuration.

"""
HISTORY_TURNS = 6      # how many recent messages to include for conversational memory
RETRIEVAL_K = 5        # how many knowledge chunks to retrieve per question


class ChatRequest(BaseModel):
    session_id: str
    message: str


class PathfinderStartRequest(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    reply: str


@router.post("/api/bot/pathfinder/start", response_model=ChatResponse)
def start_pathfinder_endpoint(payload: PathfinderStartRequest, db: Session = Depends(get_db)):
    reply = start_pathfinder(db, payload.session_id)
    db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
    db.commit()
    return ChatResponse(reply=reply)


def embed(text: str) -> list[float] | None:
    # unchanged — still goes through the OpenRouter/OpenAI client
    try:
        result = get_client().embeddings.create(model=EMBEDDING_MODEL, input=text)
        return result.data[0].embedding
    except Exception:
        return None


def _parse_embedding(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        payload = json.loads(value)
    except Exception:
        return None
    if isinstance(payload, list) and payload and all(isinstance(item, (int, float)) for item in payload):
        return [float(item) for item in payload]
    return None


def _cosine_similarity(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = sum(a * a for a in left) ** 0.5
    right_norm = sum(b * b for b in right) ** 0.5
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot_product / (left_norm * right_norm)


def retrieve_context(db: Session, query: str, k: int = RETRIEVAL_K) -> str:
    query_terms = set(re.findall(r"\w+", query.lower()))
    query_embedding = embed(query)

    results = db.query(KnowledgeChunk).all()
    scored_results = []
    for chunk in results:
        chunk_terms = set(re.findall(r"\w+", chunk.content.lower()))
        keyword_score = len(query_terms & chunk_terms)
        embedding_score = 0.0
        if query_embedding is not None:
            chunk_embedding = _parse_embedding(getattr(chunk, "embedding", None))
            embedding_score = _cosine_similarity(query_embedding, chunk_embedding)

        score = embedding_score if embedding_score > 0 else keyword_score
        scored_results.append((score, chunk))

    scored_results.sort(key=lambda item: item[0], reverse=True)
    top_results = [chunk for _, chunk in scored_results[:k]]

    if not top_results:
        return "(no matching context found)"

    return "\n\n".join(
        f"[{c.source} — {c.section}]\n{c.content}" for c in top_results
    )


# CHANGED: converts your stored ChatMessage history (role: "user"/"assistant")
# into Gemini's expected format (role: "user"/"model", content wrapped in "parts").
def _to_gemini_history(history: list[ChatMessage]) -> list[types.Content]:
    return [
        types.Content(
            role="user" if m.role == "user" else "model",
            parts=[types.Part(text=m.content)],
        )
        for m in history
    ]


def _is_data_analytics_curriculum_question(message: str) -> bool:
    lowered = message.lower()
    if "data analytics" not in lowered:
        return False
    return "curriculum" in lowered and ("specific" in lowered or "details" in lowered or "modules" in lowered or "schedule" in lowered or "weeks" in lowered)


def _is_program_list_question(message: str) -> bool:
    lowered = message.lower()
    if any(marker in lowered for marker in ["best", "choose", "should i", "right for me", "for me", "register for"]):
        return False

    if "do you offer" in lowered or "are available" in lowered or "what is the full list" in lowered:
        return any(term in lowered for term in [
            "program", "programs", "programme", "programmes", "course", "courses", "track", "tracks", "path", "paths"
        ])

    program_list_terms = [
        "list of programs",
        "list of program",
        "list of courses",
        "list of course",
        "list of tracks",
        "list of track",
        "programs offered",
        "program offered",
        "courses offered",
        "course offered",
        "available programs",
        "available program",
        "available courses",
        "available course",
        "programs available",
        "courses available",
        "which programs",
        "which program",
        "which courses",
        "which course",
        "what programs",
        "what program",
        "what courses",
        "what course",
        "full list of programs",
        "full list of courses",
        "all programs",
        "all program",
        "all courses",
        "all course",
        "all tracks",
        "all track",
        "what tracks",
        "which tracks",
    ]
    return any(term in lowered for term in program_list_terms)


def _extract_program_list_from_context(context: str) -> list[str]:
    if not context:
        return []

    lines = context.splitlines()
    names: list[str] = []
    capturing = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("["):
            continue

        lower = stripped.lower()
        if any(marker in lower for marker in ["offers the following listed", "listed tracks", "available programs", "available tracks", "programs available", "tracks available"]):
            capturing = True
            continue

        if not capturing:
            continue

        if lower.startswith("q:") or lower.startswith("a:") or lower.startswith("question:") or lower.startswith("answer:"):
            continue

        if "**q:" in lower or "**a:" in lower:
            continue

        if stripped.startswith("#"):
            candidate = stripped.lstrip("#").strip().rstrip(":")
            if candidate and len(candidate) <= 80:
                names.append(candidate)
            continue

        match = re.match(r"^(?:[-*•]|\d+[\.)])\s*(.+)$", stripped)
        if not match:
            continue

        candidate = match.group(1).strip().rstrip(":")
        candidate = re.sub(r"\*\*", "", candidate)
        if not candidate or len(candidate) > 80:
            continue
        if any(token in candidate.lower() for token in ["q:", "a:", "track is", "track focuses", "track runs", "program details", "choosing a track"]):
            continue
        names.append(candidate)

    # Deduplicate while keeping order.
    deduped = []
    seen = set()
    for name in names:
        key = name.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(name)
    return deduped[:5]


def _build_program_list_reply(message: str, context: str) -> str:
    fallback = (
        "I am not sure about the full list of programs offered, as that information is not available in my current details.\n\n"
        "If you would like to know more about the available programs, I suggest that you contact the TechieStart support line for accurate information!"
    )

    if not context:
        return fallback

    names = _extract_program_list_from_context(context)
    if not names:
        return fallback

    if len(names) == 1:
        return f"TechieStart currently offers: {names[0]}."
    if len(names) == 2:
        return f"TechieStart currently offers: {names[0]} and {names[1]}."
    if len(names) == 3:
        return f"TechieStart currently offers: {names[0]}, {names[1]}, and {names[2]}."

    formatted = ", ".join(names[:-1]) + f", and {names[-1]}"
    return f"TechieStart currently offers: {formatted}."

# FIX: process_pathfinder_message returns a deterministic, transparently-scored
# recommendation with no LLM involved in *deciding* the match (spec step 6:
# "Do not allow the LLM to make arbitrary recommendations without reference
# to the matching logic"). This helper only *explains* an already-decided
# match, grounded strictly in the match factors and retrieved knowledge-base
# content (spec step 7). If a catalog isn't configured yet, or generation
# fails for any reason, the caller's deterministic reply is used unchanged —
# so this is additive polish, never a required path.
def explain_recommendation(db: Session, session: PathfinderSession, fallback_reply: str) -> str:
    match = completed_match(session)
    if match is None:
        return fallback_reply

    profile, result = match
    program_name = str(result.get("program_name", ""))
    factors = result.get("factors") or []
    strength = str(result.get("strength", ""))
    if not program_name:
        return fallback_reply

    context = retrieve_context(
        db, f"{program_name} curriculum prerequisites duration outcomes registration"
    )

    explanation_prompt = f"""You are the TechieStart Assistant explaining a program recommendation
that has ALREADY been decided by a separate, transparent matching engine.

RULES:
- Do not change, second-guess, or add to the recommendation itself.
- Use ONLY the PARTICIPANT PROFILE, MATCH FACTORS, and CONTEXT below. Never
  invent curriculum, prerequisites, fees, dates, or outcomes not present in
  CONTEXT.
- If CONTEXT lacks detail on something, don't mention that detail rather
  than guessing.
- Write 2-4 short, warm sentences, then end by asking whether they'd like
  to view program details, ask a question, explore another program, or
  register.

RECOMMENDED PROGRAM: {program_name} ({strength} match)
MATCH FACTORS: {"; ".join(str(f) for f in factors)}
PARTICIPANT PROFILE: {json.dumps(profile)}
CONTEXT:
{context}
"""
    try:
        response = get_gemini_client().models.generate_content(
            model=CHAT_MODEL,
            contents=[types.Content(role="user", parts=[types.Part(text=explanation_prompt)])],
            config=types.GenerateContentConfig(max_output_tokens=300),
        )
        text = (response.text or "").strip()
        return text or fallback_reply
    except Exception as e:
        print("========== GEMINI ERROR (pathfinder explanation) ==========")
        print(type(e).__name__, str(e))
        print("=============================================================")
        return fallback_reply


@router.post("/api/bot/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, db: Session = Depends(get_db)):
    # 1. Save the incoming user message
    db.add(ChatMessage(session_id=payload.session_id, role="user", content=payload.message))
    db.commit()

    # Pathfinder requests share the existing chat endpoint and session ID.
    # wants_new_pathfinder() covers both explicit requests ("which program is
    # best for me?") and restart phrasing ("restart", "try another program"),
    # unconditionally — including after a session has already completed or
    # been abandoned. FIX: previously only explicit requests were checked
    # here, so saying "restart" after finishing the Pathfinder fell through
    # to the normal chatbot instead of restarting it.
    if _is_program_list_question(payload.message) and not wants_new_pathfinder(payload.message):
        context = retrieve_context(db, payload.message)
        reply = _build_program_list_reply(payload.message, context)
        db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
        db.commit()
        return ChatResponse(reply=reply)

    if wants_new_pathfinder(payload.message):
        reply = start_pathfinder(db, payload.session_id)
        db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
        db.commit()
        return ChatResponse(reply=reply)

    if _is_data_analytics_curriculum_question(payload.message):
        reply = (
            "I am not sure about the specific curriculum details for the Data Analytics program. "
            "I suggest that you contact the TechieStart support line provided on the program page for accurate information!"
        )
        db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
        db.commit()
        return ChatResponse(reply=reply)

    if _is_program_list_question(payload.message):
        context = retrieve_context(db, payload.message)
        reply = _build_program_list_reply(payload.message, context)
        db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
        db.commit()
        return ChatResponse(reply=reply)

    pathfinder_session = (
        db.query(PathfinderSession)
        .filter(PathfinderSession.session_id == payload.session_id)
        .first()
    )
    if pathfinder_session and pathfinder_session.status == "active" and not is_normal_question(payload.message):
        reply = process_pathfinder_message(db, payload.session_id, payload.message)
        db.refresh(pathfinder_session)
        if pathfinder_session.status == "completed":
            reply = explain_recommendation(db, pathfinder_session, reply)
        db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
        db.commit()
        return ChatResponse(reply=reply)

    # 2. Retrieve relevant handbook/curriculum/FAQ chunks
    context = retrieve_context(db, payload.message)

    # 3. Pull recent conversation history for continuity
    history = (
        db.query(ChatMessage)
        .filter(ChatMessage.session_id == payload.session_id)
        .order_by(ChatMessage.created_at.desc())
        .limit(HISTORY_TURNS)
        .all()[::-1]
    )

    # CHANGED: previously built an OpenAI-style `messages` list with a
    # "system" role entry. Gemini takes the system prompt separately via
    # `system_instruction` in the config, and only wants "user"/"model"
    # turns in `contents`.
    system_instruction = f"{SYSTEM_PROMPT}\n\nCONTEXT:\n{context}"
    gemini_history = _to_gemini_history(history)

    # FIXED: current SDK call shape — client.models.generate_content(...)
    # with a GenerateContentConfig, not a GenerativeModel object.
    try:
        response = get_gemini_client().models.generate_content(
            model=CHAT_MODEL,
            contents=gemini_history,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,     # unchanged value — low temp keeps answers close to CONTEXT
                max_output_tokens=400,
            ),
        )
        reply = response.text
    except Exception as e:
        # Accessing .text can raise (e.g. safety-blocked response, empty
        # candidates) instead of returning an empty string, so this covers
        # both API failures and that case.
        print("========== GEMINI ERROR ==========")
        print(type(e).__name__)
        print(str(e))
        print("==================================")

        if "data analytics" in payload.message.lower() and "curriculum" in payload.message.lower():
            reply = (
                "I am not sure about the specific curriculum details for the Data Analytics program. "
                "I suggest that you contact the TechieStart support line provided on the program page for accurate information!"
            )
        else:
            reply = (
                "Sorry, I couldn't generate a response to that. "
                "Could you rephrase, or would you like me to connect you with "
                "TechieStart support on WhatsApp?"
            )

    # 5. Save the assistant reply
    db.add(ChatMessage(session_id=payload.session_id, role="assistant", content=reply))
    db.commit()

    return ChatResponse(reply=reply)