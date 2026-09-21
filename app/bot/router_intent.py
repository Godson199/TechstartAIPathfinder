"""
TechieStart conversation intent router.

This module does NOT answer the user.

It determines what the user's latest message means
while a Pathfinder session is active.

Possible intents:

- pathfinder_answer
- rag_question
- both
- restart
- casual
- unclear
"""

import json
import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()


CHAT_MODEL = "gemini-flash-lite-latest"


# ============================================================
# GEMINI CLIENT
# ============================================================

_gemini_client: genai.Client | None = None


def get_router_gemini_client() -> genai.Client:

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
# ROUTER PROMPT
# ============================================================

ROUTER_PROMPT = """
You are the conversation router for the TechieStart chatbot.

The user may currently be going through the TechieStart Pathfinder.

The Pathfinder asks questions one at a time to understand the user's
interests, experience, goals, activities, preferred work style, and
learning preferences.

Your job is NOT to answer the user's question.

Your job is ONLY to classify the user's latest message.

AVAILABLE INTENTS:

1. "pathfinder_answer"

Use this when the user is answering the current Pathfinder question.

Example:

Current Pathfinder question:
"What area of technology interests you most right now?"

User:
"I am interested in artificial intelligence and machine learning."

Classification:
pathfinder_answer

--------------------------------------------------

2. "rag_question"

Use this when the user is asking a TechieStart information/FAQ/curriculum
question instead of answering the current Pathfinder question.

Examples:

"How much is the registration fee?"

"How long does the program last?"

"Do I need a Google account?"

"What does the Data Analytics curriculum cover?"

"How do I get my certificate?"

IMPORTANT:
If this happens while Pathfinder is active, DO NOT advance the Pathfinder.

--------------------------------------------------

3. "both"

Use this when the user does BOTH things in one message:

A. provides an answer to the current Pathfinder question
AND
B. asks a separate TechieStart information question.

Example:

Current Pathfinder question:
"How would you describe your current technical experience?"

User:
"I am a complete beginner. Also, does the AI and ML program require
previous experience?"

Classification:
both

pathfinder_answer:
"I am a complete beginner."

rag_question:
"Does the AI and ML program require previous experience?"

--------------------------------------------------

4. "restart"

Use this only when the user clearly wants to restart the Pathfinder.

Examples:

"restart"

"start over"

"begin again"

"I want to try another program"

"Let's start the assessment again"

Do NOT classify a normal Pathfinder answer as restart merely because it
contains words such as "program", "course", "track", or "path".

--------------------------------------------------

5. "casual"

Use this for simple conversational messages that do not contain a
TechieStart information question and are not an answer to the Pathfinder.

Examples:

"hello"

"thanks"

"okay"

"that's nice"

"good morning"

If Pathfinder is active, the application will answer the casual message
and then restore the pending Pathfinder question.

--------------------------------------------------

6. "unclear"

Use this only when the meaning genuinely cannot be determined.

IMPORTANT RULES:

- Do not advance Pathfinder for a rag_question.
- Do not invent a Pathfinder answer.
- Do not turn a TechieStart information question into a Pathfinder answer.
- A user's answer may be a full sentence or multiple sentences.
- A Pathfinder answer may contain question-like wording.
- Use the CURRENT PATHFINDER QUESTION to determine whether the message
  answers it.
- If a message contains both a Pathfinder answer and a TechieStart
  question, use "both".
- Extract the Pathfinder answer separately.
- Extract the TechieStart question separately.
- Do not answer either question.

Return ONLY valid JSON in exactly this structure:

{
  "intent": "pathfinder_answer | rag_question | both | restart | casual | unclear",
  "pathfinder_answer": null,
  "rag_question": null
}

If the intent is "pathfinder_answer":

{
  "intent": "pathfinder_answer",
  "pathfinder_answer": "the extracted answer",
  "rag_question": null
}

If the intent is "rag_question":

{
  "intent": "rag_question",
  "pathfinder_answer": null,
  "rag_question": "the extracted TechieStart question"
}

If the intent is "both":

{
  "intent": "both",
  "pathfinder_answer": "the part answering the Pathfinder",
  "rag_question": "the TechieStart question"
}

If the intent is "casual", "restart", or "unclear":

{
  "intent": "casual",
  "pathfinder_answer": null,
  "rag_question": null
}

CURRENT PATHFINDER QUESTION:
{current_question}

PREVIOUS PATHFINDER ANSWERS:
{answers}

USER'S LATEST MESSAGE:
{message}
"""


# ============================================================
# SAFE JSON EXTRACTION
# ============================================================

def _safe_json(text: str) -> dict:

    if not text:
        return {
            "intent": "unclear",
            "pathfinder_answer": None,
            "rag_question": None,
        }

    text = text.strip()

    try:
        result = json.loads(text)

        if isinstance(result, dict):
            return result

    except json.JSONDecodeError:
        pass

    # Sometimes an API/model can wrap JSON in markdown.
    if "{" in text and "}" in text:

        start = text.find("{")
        end = text.rfind("}") + 1

        candidate = text[start:end]

        try:
            result = json.loads(candidate)

            if isinstance(result, dict):
                return result

        except json.JSONDecodeError:
            pass

    return {
        "intent": "unclear",
        "pathfinder_answer": None,
        "rag_question": None,
    }


# ============================================================
# NORMALIZE RESULT
# ============================================================

VALID_INTENTS = {
    "pathfinder_answer",
    "rag_question",
    "both",
    "restart",
    "casual",
    "unclear",
}


def _normalize_result(
    result: dict,
    original_message: str,
) -> dict:

    intent = str(
        result.get(
            "intent",
            "unclear",
        )
    ).strip().lower()

    if intent not in VALID_INTENTS:
        intent = "unclear"

    pathfinder_answer = result.get(
        "pathfinder_answer"
    )

    rag_question = result.get(
        "rag_question"
    )

    if pathfinder_answer is not None:

        pathfinder_answer = str(
            pathfinder_answer
        ).strip()

        if not pathfinder_answer:
            pathfinder_answer = None

    if rag_question is not None:

        rag_question = str(
            rag_question
        ).strip()

        if not rag_question:
            rag_question = None

    # Safety fallback:
    #
    # If Gemini says "pathfinder_answer" but did not
    # extract anything, use the original message.
    if (
        intent == "pathfinder_answer"
        and not pathfinder_answer
    ):
        pathfinder_answer = original_message.strip()

    # Same for a RAG question.
    if (
        intent == "rag_question"
        and not rag_question
    ):
        rag_question = original_message.strip()

    # Same for both.
    if intent == "both":

        if not pathfinder_answer:
            pathfinder_answer = original_message.strip()

        if not rag_question:
            rag_question = original_message.strip()

    return {
        "intent": intent,
        "pathfinder_answer": pathfinder_answer,
        "rag_question": rag_question,
    }


# ============================================================
# CLASSIFY MESSAGE
# ============================================================

def classify_message(
    message: str,
    current_question: str,
    answers: dict,
) -> dict:

    prompt = ROUTER_PROMPT.format(
        current_question=current_question,
        answers=json.dumps(
            answers,
            ensure_ascii=False,
        ),
        message=message,
    )

    try:

        response = (
            get_router_gemini_client()
            .models.generate_content(
                model=CHAT_MODEL,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part(
                                text=prompt
                            )
                        ],
                    )
                ],
                config=types.GenerateContentConfig(
                    temperature=0,
                    max_output_tokens=250,
                    response_mime_type="application/json",
                ),
            )
        )

        result = _safe_json(
            response.text or ""
        )

        return _normalize_result(
            result,
            message,
        )

    except Exception as exc:

        print(
            "========== GEMINI ROUTER ERROR =========="
        )
        print(
            type(exc).__name__,
            str(exc),
        )
        print(
            "=========================================="
        )

        # Do not accidentally advance Pathfinder
        # when the classifier fails.
        #
        # "unclear" is deliberately safe.
        return {
            "intent": "unclear",
            "pathfinder_answer": None,
            "rag_question": None,
        }