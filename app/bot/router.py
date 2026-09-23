"""
TechieStart chatbot endpoint.

Architecture:

    /api/bot/pathfinder/start
            |
            v
       pathfinder.py
            |
            v
    /api/bot/chat
            |
            v
    router_intent.py
            |
       +----+----+
       |         |
       v         v
   Pathfinder   RAG
       |         |
       +----+----+
            |
            v
       final reply

The Pathfinder state is never advanced unless router_intent.py
determines that the user actually answered the current Pathfinder
question.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from fastapi import APIRouter, Depends
from openai import OpenAI
from pydantic import BaseModel
from sqlalchemy.orm import Session

from google import genai
from google.genai import types

from app.core.database import get_db
from app.models.chat import ChatMessage
from app.models.knowledge import KnowledgeChunk
from app.models.pathfinder import PathfinderSession

from app.bot.pathfinder import (
    QUESTIONS,
    completed_match,
    get_active_pathfinder_state,
    is_explicit_restart,
    process_pathfinder_message,
    start_pathfinder,
    wants_new_pathfinder,
)

from app.bot.router_intent import (
    classify_message,
)


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv(
    Path(__file__).resolve().parents[2]
    / ".env"
)


router = APIRouter()


# ============================================================
# OPENAI / OPENROUTER CLIENT
#
# Used ONLY for embeddings.
# ============================================================

def get_client() -> OpenAI:

    api_key = (
        os.getenv("API_KEY")
        or os.getenv("OPENAI_API_KEY")
    )

    if not api_key:

        raise RuntimeError(
            "Missing API key. Set API_KEY or "
            "OPENAI_API_KEY in your .env file."
        )

    return OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://openrouter.ai/api/v1",
        ),
    )


# ============================================================
# GEMINI CLIENT
# ============================================================

_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:

    global _gemini_client

    if _gemini_client is None:

        api_key = os.getenv(
            "GEMINI_API_KEY"
        )

        if not api_key:

            raise RuntimeError(
                "Missing API key. Set GEMINI_API_KEY "
                "in your .env file."
            )

        _gemini_client = genai.Client(
            api_key=api_key
        )

    return _gemini_client


# ============================================================
# MODELS
# ============================================================

CHAT_MODEL = "gemini-flash-lite-latest"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "text-embedding-3-small",
)

HISTORY_TURNS = 6

RETRIEVAL_K = 5


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are the TechieStart Assistant.

Your job is to help users with questions about TechieStart while also
maintaining a friendly, natural conversation.

IMPORTANT:

The application may temporarily route a user's message to RAG while
the user is inside the Pathfinder.

When that happens:

- Answer ONLY the TechieStart information question.
- Do NOT continue the Pathfinder yourself.
- Do NOT answer the Pathfinder question.
- Do NOT invent a Pathfinder answer.
- Do NOT recommend a program unless the application explicitly asks you
  to explain an already-determined recommendation.
- The application will restore the pending Pathfinder question after
  your answer.

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
registration, requirements, instructors, services, curriculum, or any
other specific TechieStart information, answer ONLY using the CONTEXT
provided below.

The CONTEXT comes from the official TechieStart FAQ, participant
handbook, curriculum, and program information.

3. MIXED QUESTIONS

If a message contains both casual conversation and a TechieStart
question, respond naturally to the casual part and answer the
TechieStart question using ONLY the CONTEXT.

4. IF INFORMATION IS NOT IN THE CONTEXT

If a user asks a TechieStart-related question and the answer cannot
be found in the CONTEXT, do not guess, assume, or invent information.

Instead, politely say that you are not sure and suggest that the user
contact the TechieStart support line for accurate information.

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

If the requested TechieStart information is not available in the
CONTEXT, you may also suggest continuing with the TechieStart support
team on WhatsApp.

Do not invent or guess the WhatsApp number. Only provide the WhatsApp
link or number if it is explicitly provided in the CONTEXT or
application configuration.
"""


# ============================================================
# API SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    session_id: str
    message: str


class PathfinderStartRequest(BaseModel):
    session_id: str


class ChatResponse(BaseModel):
    reply: str


# ============================================================
# EMBEDDINGS
# ============================================================

def embed(
    text: str,
) -> list[float] | None:

    try:

        result = (
            get_client()
            .embeddings.create(
                model=EMBEDDING_MODEL,
                input=text,
            )
        )

        return result.data[0].embedding

    except Exception as exc:

        print(
            "Embedding error:",
            type(exc).__name__,
            str(exc),
        )

        return None


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def _parse_embedding(
    value: str | None,
) -> list[float] | None:

    if not value:
        return None

    try:

        payload = json.loads(value)

    except Exception:

        return None

    if (
        isinstance(payload, list)
        and payload
        and all(
            isinstance(
                item,
                (int, float),
            )
            for item in payload
        )
    ):

        return [
            float(item)
            for item in payload
        ]

    return None


def _cosine_similarity(
    left: list[float] | None,
    right: list[float] | None,
) -> float:

    if not left or not right:
        return 0.0

    if len(left) != len(right):
        return 0.0

    dot_product = sum(
        a * b
        for a, b in zip(left, right)
    )

    left_norm = (
        sum(a * a for a in left)
        ** 0.5
    )

    right_norm = (
        sum(b * b for b in right)
        ** 0.5
    )

    if (
        left_norm == 0
        or right_norm == 0
    ):
        return 0.0

    return dot_product / (
        left_norm * right_norm
    )


# ============================================================
# RAG RETRIEVAL
# ============================================================

def retrieve_context(
    db: Session,
    query: str,
    k: int = RETRIEVAL_K,
) -> str:

    query_terms = set(
        re.findall(
            r"\w+",
            query.lower(),
        )
    )

    query_embedding = embed(query)

    results = (
        db.query(KnowledgeChunk)
        .all()
    )

    scored_results = []

    for chunk in results:

        chunk_terms = set(
            re.findall(
                r"\w+",
                chunk.content.lower(),
            )
        )

        keyword_score = len(
            query_terms
            & chunk_terms
        )

        embedding_score = 0.0

        if query_embedding is not None:

            chunk_embedding = (
                _parse_embedding(
                    getattr(
                        chunk,
                        "embedding",
                        None,
                    )
                )
            )

            embedding_score = (
                _cosine_similarity(
                    query_embedding,
                    chunk_embedding,
                )
            )

        score = (
            embedding_score
            if embedding_score > 0
            else keyword_score
        )

        scored_results.append(
            (
                score,
                chunk,
            )
        )

    scored_results.sort(
        key=lambda item: item[0],
        reverse=True,
    )

    top_results = [
        chunk
        for _, chunk
        in scored_results[:k]
    ]

    if not top_results:

        return "(no matching context found)"

    return "\n\n".join(
        (
            f"[{c.source} — {c.section}]\n"
            f"{c.content}"
        )
        for c in top_results
    )


def retrieve_program_list_context(
    db: Session,
    query: str,
) -> str:

    return retrieve_context(
        db,
        f"{query} TechieStart available tracks programme list",
    )


# ============================================================
# GEMINI HISTORY
# ============================================================

def _to_gemini_history(
    history: list[ChatMessage],
) -> list[types.Content]:

    return [
        types.Content(
            role=(
                "user"
                if message.role == "user"
                else "model"
            ),
            parts=[
                types.Part(
                    text=message.content
                )
            ],
        )
        for message in history
    ]


# ============================================================
# PROGRAM LIST DETECTION
# ============================================================

def _is_program_list_question(
    message: str,
) -> bool:

    lowered = message.lower()

    # Do not treat recommendation requests as simple
    # program-list requests.
    if any(
        marker in lowered
        for marker in [
            "best",
            "choose",
            "should i",
            "right for me",
            "for me",
            "register for",
        ]
    ):
        return False

    if (
        "do you offer" in lowered
        or "are available" in lowered
        or "what is the full list" in lowered
    ):

        return any(
            term in lowered
            for term in [
                "program",
                "programs",
                "programme",
                "programmes",
                "course",
                "courses",
                "track",
                "tracks",
                "path",
                "paths",
            ]
        )

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

    return any(
        term in lowered
        for term in program_list_terms
    )


# ============================================================
# PROGRAM LIST EXTRACTION
# ============================================================

def _extract_program_list_from_context(
    context: str,
) -> list[str]:

    if not context:
        return []

    lines = context.splitlines()

    names: list[str] = []

    capturing = False

    def add_candidate(value: str) -> None:
        candidate = value.strip().rstrip(":")
        candidate = re.sub(r"\*\*", "", candidate)
        candidate = re.split(
            r"\s+(?:each|the|this|these|all|participants)\b",
            candidate,
            maxsplit=1,
            flags=re.IGNORECASE,
        )[0].strip().rstrip(":")

        if (
            not candidate
            or len(candidate) > 80
            or any(
                token in candidate.lower()
                for token in [
                    "q:",
                    "a:",
                    "track is",
                    "track focuses",
                    "track runs",
                    "program details",
                    "choosing a track",
                ]
            )
        ):
            return

        names.append(candidate)

    for line in lines:

        stripped = line.strip()

        if not stripped:
            continue

        if stripped.startswith("["):
            continue

        lower = stripped.lower()

        if any(
            marker in lower
            for marker in [
                "offers the following listed",
                "listed tracks",
                "available programs",
                "available tracks",
                "programs available",
                "tracks available",
            ]
        ) or (
            any(
                term in lower
                for term in [
                    "program",
                    "programme",
                    "course",
                    "track",
                    "path",
                ]
            )
            and any(
                marker in lower
                for marker in [
                    "offer",
                    "available",
                    "include",
                    "following",
                ]
            )
        ):

            capturing = True

        if stripped.startswith("#"):
            continue

        inline_items = re.findall(
            r"(?:^|\s)(?:\d+[.)]|[-*])\s+(.+?)(?=\s+(?:\d+[.)]|[-*])\s+|$)",
            stripped,
        )

        if capturing and inline_items:
            for item in inline_items:
                add_candidate(item)
            continue

        if not capturing:
            continue

        if (
            lower.startswith("q:")
            or lower.startswith("a:")
            or lower.startswith("question:")
            or lower.startswith("answer:")
        ):
            continue

        if (
            "**q:" in lower
            or "**a:" in lower
        ):
            continue

        match = re.match(
            r"^(?:[-*•]|\d+[.)])\s*(.+)$",
            stripped,
        )

        if not match:
            continue

        add_candidate(match.group(1))

    # Deduplicate while preserving order.
    deduped = []

    seen = set()

    for name in names:

        key = name.lower()

        if key not in seen:

            seen.add(key)
            deduped.append(name)

    return deduped[:5]


# ============================================================
# PROGRAM LIST RESPONSE
# ============================================================

def _build_program_list_reply(
    message: str,
    context: str,
) -> str:

    fallback = (
        "I am not sure about the full list of programs "
        "offered, as that information is not available "
        "in my current details.\n\n"
        "If you would like to know more about the available "
        "programs, I suggest that you contact the TechieStart "
        "support line for accurate information!"
    )

    if not context:
        return fallback

    names = (
        _extract_program_list_from_context(
            context
        )
    )

    if not names:
        return fallback

    if len(names) == 1:

        return (
            f"TechieStart currently offers: "
            f"{names[0]}."
        )

    if len(names) == 2:

        return (
            f"TechieStart currently offers: "
            f"{names[0]} and {names[1]}."
        )

    if len(names) == 3:

        return (
            f"TechieStart currently offers: "
            f"{names[0]}, {names[1]}, "
            f"and {names[2]}."
        )

    formatted = (
        ", ".join(names[:-1])
        + f", and {names[-1]}"
    )

    return (
        f"TechieStart currently offers: "
        f"{formatted}."
    )


# ============================================================
# NORMAL RAG ANSWER
# ============================================================

def answer_rag_question(
    db: Session,
    session_id: str,
    question: str,
) -> str:

    context = retrieve_context(
        db,
        question,
    )

    history = (
        db.query(ChatMessage)
        .filter(
            ChatMessage.session_id
            == session_id
        )
        .order_by(
            ChatMessage.created_at.desc()
        )
        .limit(HISTORY_TURNS)
        .all()[::-1]
    )

    system_instruction = (
        f"{SYSTEM_PROMPT}\n\n"
        f"CONTEXT:\n{context}"
    )

    gemini_history = (
        _to_gemini_history(history)
    )

    try:

        response = (
            get_gemini_client()
            .models.generate_content(
                model=CHAT_MODEL,
                contents=gemini_history,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=400,
                ),
            )
        )

        reply = (
            response.text or ""
        ).strip()

        if reply:
            return reply

        return (
            "I couldn't find enough information "
            "to answer that accurately. "
            "Please contact TechieStart support "
            "for assistance."
        )

    except Exception as exc:

        print(
            "========== GEMINI ERROR =========="
        )
        print(
            type(exc).__name__
        )
        print(
            str(exc)
        )
        print(
            "=================================="
        )

        return (
            "Sorry, I couldn't generate a response "
            "to that right now. Could you rephrase "
            "your question, or would you like me to "
            "connect you with TechieStart support "
            "on WhatsApp?"
        )


# ============================================================
# RECOMMENDATION EXPLANATION
# ============================================================

def explain_recommendation(
    db: Session,
    session: PathfinderSession,
    fallback_reply: str,
) -> str:

    match = completed_match(
        session
    )

    if match is None:
        return fallback_reply

    profile, result = match

    program_name = str(
        result.get(
            "program_name",
            "",
        )
    )

    factors = (
        result.get("factors")
        or []
    )

    strength = str(
        result.get(
            "strength",
            "",
        )
    )

    if not program_name:
        return fallback_reply

    context = retrieve_context(
        db,
        (
            f"{program_name} curriculum "
            f"prerequisites duration outcomes "
            f"registration"
        ),
    )

    explanation_prompt = f"""
You are the TechieStart Assistant explaining a program recommendation
that has ALREADY been decided by a separate, transparent matching engine.

RULES:

- Do not change, second-guess, or add to the recommendation itself.
- Use ONLY the PARTICIPANT PROFILE, MATCH FACTORS, and CONTEXT below.
- Never invent curriculum, prerequisites, fees, dates, or outcomes.
- If CONTEXT lacks detail on something, do not mention that detail.
- Write 2-4 short, warm sentences.
- End by asking whether they would like to view program details, ask a
  question, explore another program, or register.

RECOMMENDED PROGRAM:
{program_name} ({strength} match)

MATCH FACTORS:
{"; ".join(str(f) for f in factors)}

PARTICIPANT PROFILE:
{json.dumps(profile, ensure_ascii=False)}

CONTEXT:
{context}
"""

    try:

        response = (
            get_gemini_client()
            .models.generate_content(
                model=CHAT_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=explanation_prompt
                            )
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    max_output_tokens=300,
                ),
            )
        )

        text = (
            response.text or ""
        ).strip()

        return text or fallback_reply

    except Exception as exc:

        print(
            "========== GEMINI ERROR "
            "(pathfinder explanation) =========="
        )

        print(
            type(exc).__name__,
            str(exc),
        )

        print(
            "========================================"
        )

        return fallback_reply


# ============================================================
# START PATHFINDER ENDPOINT
# ============================================================

@router.post(
    "/api/bot/pathfinder/start",
    response_model=ChatResponse,
)
def start_pathfinder_endpoint(
    payload: PathfinderStartRequest,
    db: Session = Depends(get_db),
):

    reply = start_pathfinder(
        db,
        payload.session_id,
    )

    db.add(
        ChatMessage(
            session_id=payload.session_id,
            role="assistant",
            content=reply,
        )
    )

    db.commit()

    return ChatResponse(
        reply=reply
    )


# ============================================================
# MAIN CHAT ENDPOINT
# ============================================================

@router.post(
    "/api/bot/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    db: Session = Depends(get_db),
):

    message = payload.message.strip()

    if not message:

        return ChatResponse(
            reply=(
                "Please send me a message and "
                "I'll be happy to help."
            )
        )

    # ========================================================
    # 1. SAVE USER MESSAGE
    # ========================================================

    db.add(
        ChatMessage(
            session_id=payload.session_id,
            role="user",
            content=message,
        )
    )

    db.commit()

    # ========================================================
    # 2. GET CURRENT PATHFINDER STATE
    # ========================================================

    pathfinder_state = (
        get_active_pathfinder_state(
            db,
            payload.session_id,
        )
    )

    # ========================================================
    # 3. EXPLICIT RESTART
    #
    # This is checked before classification.
    # "restart", "start over", etc. should always restart.
    # ========================================================

    if is_explicit_restart(message):

        reply = start_pathfinder(
            db,
            payload.session_id,
        )

        db.add(
            ChatMessage(
                session_id=payload.session_id,
                role="assistant",
                content=reply,
            )
        )

        db.commit()

        return ChatResponse(
            reply=reply
        )

    # ========================================================
    # 4. ACTIVE PATHFINDER
    #
    # THIS IS THE MAIN FIX.
    #
    # We DO NOT immediately pass the message to
    # process_pathfinder_message().
    #
    # First we ask router_intent.py what the message means.
    # ========================================================

    if pathfinder_state:

        current_question = (
            pathfinder_state[
                "current_question"
            ]
        )

        answers = (
            pathfinder_state[
                "answers"
            ]
        )

        classification = classify_message(
            message=message,
            current_question=current_question.prompt,
            answers=answers,
        )

        intent = classification[
            "intent"
        ]

        print(
            "========== PATHFINDER ROUTER =========="
        )

        print(
            "Intent:",
            intent,
        )

        print(
            "Pathfinder answer:",
            classification.get(
                "pathfinder_answer"
            ),
        )

        print(
            "RAG question:",
            classification.get(
                "rag_question"
            ),
        )

        print(
            "========================================"
        )

        # ====================================================
        # 4A. REAL PATHFINDER ANSWER
        # ====================================================

        if intent == "pathfinder_answer":

            pathfinder_answer = (
                classification.get(
                    "pathfinder_answer"
                )
                or message
            )

            reply = process_pathfinder_message(
                db,
                payload.session_id,
                pathfinder_answer,
            )

            # Check whether Pathfinder has just completed.
            updated_state = (
                get_active_pathfinder_state(
                    db,
                    payload.session_id,
                )
            )

            if updated_state is None:

                pathfinder_session = (
                    db.query(
                        PathfinderSession
                    )
                    .filter(
                        PathfinderSession.session_id
                        == payload.session_id
                    )
                    .first()
                )

                if (
                    pathfinder_session
                    and pathfinder_session.status
                    == "completed"
                ):

                    reply = (
                        explain_recommendation(
                            db,
                            pathfinder_session,
                            reply,
                        )
                    )

            db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    role="assistant",
                    content=reply,
                )
            )

            db.commit()

            return ChatResponse(
                reply=reply
            )

        # ====================================================
        # 4B. RAG QUESTION ONLY
        #
        # IMPORTANT:
        # Pathfinder is NOT advanced.
        # ====================================================

        if intent == "rag_question":

            rag_question = (
                classification.get(
                    "rag_question"
                )
                or message
            )

            # Special program-list handling.
            if _is_program_list_question(
                rag_question
            ):

                context = retrieve_program_list_context(
                    db,
                    rag_question,
                )

                rag_reply = (
                    _build_program_list_reply(
                        rag_question,
                        context,
                    )
                )

            else:

                rag_reply = answer_rag_question(
                    db,
                    payload.session_id,
                    rag_question,
                )

            # Re-fetch Pathfinder state.
            #
            # It should still be active and unchanged.
            pending_state = (
                get_active_pathfinder_state(
                    db,
                    payload.session_id,
                )
            )

            if pending_state:

                pending_question = (
                    pending_state[
                        "current_question"
                    ].prompt
                )

                reply = (
                    f"{rag_reply}\n\n"
                    f"{pending_question}"
                )

            else:

                reply = rag_reply

            db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    role="assistant",
                    content=reply,
                )
            )

            db.commit()

            return ChatResponse(
                reply=reply
            )

        # ====================================================
        # 4C. BOTH
        #
        # User answered Pathfinder AND asked RAG question.
        # ====================================================

        if intent == "both":

            pathfinder_answer = (
                classification.get(
                    "pathfinder_answer"
                )
                or message
            )

            rag_question = (
                classification.get(
                    "rag_question"
                )
                or message
            )

            # -----------------------------------------------
            # FIRST: save Pathfinder answer and advance it
            # -----------------------------------------------

            pathfinder_reply = (
                process_pathfinder_message(
                    db,
                    payload.session_id,
                    pathfinder_answer,
                )
            )

            # -----------------------------------------------
            # SECOND: answer RAG question
            # -----------------------------------------------

            if _is_program_list_question(
                rag_question
            ):

                context = retrieve_program_list_context(
                    db,
                    rag_question,
                )

                rag_reply = (
                    _build_program_list_reply(
                        rag_question,
                        context,
                    )
                )

            else:

                rag_reply = answer_rag_question(
                    db,
                    payload.session_id,
                    rag_question,
                )

            # -----------------------------------------------
            # THIRD: determine Pathfinder state AFTER
            # processing the answer
            # -----------------------------------------------

            pending_state = (
                get_active_pathfinder_state(
                    db,
                    payload.session_id,
                )
            )

            if pending_state:

                next_question = (
                    pending_state[
                        "current_question"
                    ].prompt
                )

                reply = (
                    f"{rag_reply}\n\n"
                    f"{next_question}"
                )

            else:

                # Pathfinder completed.
                pathfinder_session = (
                    db.query(
                        PathfinderSession
                    )
                    .filter(
                        PathfinderSession.session_id
                        == payload.session_id
                    )
                    .first()
                )

                if (
                    pathfinder_session
                    and pathfinder_session.status
                    == "completed"
                ):

                    recommendation = (
                        explain_recommendation(
                            db,
                            pathfinder_session,
                            pathfinder_reply,
                        )
                    )

                else:

                    recommendation = (
                        pathfinder_reply
                    )

                reply = (
                    f"{rag_reply}\n\n"
                    f"{recommendation}"
                )

            db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    role="assistant",
                    content=reply,
                )
            )

            db.commit()

            return ChatResponse(
                reply=reply
            )

        # ====================================================
        # 4D. CASUAL
        #
        # Do NOT advance Pathfinder.
        # Answer casually, then restore the question.
        # ====================================================

        if intent == "casual":

            rag_reply = answer_rag_question(
                db,
                payload.session_id,
                message,
            )

            pending_state = (
                get_active_pathfinder_state(
                    db,
                    payload.session_id,
                )
            )

            if pending_state:

                reply = (
                    f"{rag_reply}\n\n"
                    f"{pending_state['current_question'].prompt}"
                )

            else:

                reply = rag_reply

            db.add(
                ChatMessage(
                    session_id=payload.session_id,
                    role="assistant",
                    content=reply,
                )
            )

            db.commit()

            return ChatResponse(
                reply=reply
            )

        # ====================================================
        # 4E. UNCLEAR
        #
        # SAFETY RULE:
        # NEVER advance Pathfinder when classification fails.
        #
        # Treat it as RAG and restore the Pathfinder question.
        # ====================================================

        rag_reply = answer_rag_question(
            db,
            payload.session_id,
            message,
        )

        pending_state = (
            get_active_pathfinder_state(
                db,
                payload.session_id,
            )
        )

        if pending_state:

            reply = (
                f"{rag_reply}\n\n"
                f"{pending_state['current_question'].prompt}"
            )

        else:

            reply = rag_reply

        db.add(
            ChatMessage(
                session_id=payload.session_id,
                role="assistant",
                content=reply,
            )
        )

        db.commit()

        return ChatResponse(
            reply=reply
        )

    # ========================================================
    # 5. NO ACTIVE PATHFINDER
    #
    # Normal chatbot mode.
    # ========================================================

    # Explicit Pathfinder request.
    if wants_new_pathfinder(message):

        reply = start_pathfinder(
            db,
            payload.session_id,
        )

        db.add(
            ChatMessage(
                session_id=payload.session_id,
                role="assistant",
                content=reply,
            )
        )

        db.commit()

        return ChatResponse(
            reply=reply
        )

    # Program list.
    if _is_program_list_question(
        message
    ):

        context = retrieve_program_list_context(
            db,
            message,
        )

        reply = (
            _build_program_list_reply(
                message,
                context,
            )
        )

        db.add(
            ChatMessage(
                session_id=payload.session_id,
                role="assistant",
                content=reply,
            )
        )

        db.commit()

        return ChatResponse(
            reply=reply
        )

    # ========================================================
    # NORMAL RAG CHAT
    # ========================================================

    reply = answer_rag_question(
        db,
        payload.session_id,
        message,
    )

    db.add(
        ChatMessage(
            session_id=payload.session_id,
            role="assistant",
            content=reply,
        )
    )

    db.commit()

    return ChatResponse(
        reply=reply
    )