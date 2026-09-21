"""Stateful, explainable TechieStart program matching."""
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from sqlalchemy.orm import Session

from app.models.pathfinder import PathfinderSession


@dataclass(frozen=True)
class Question:
    key: str
    prompt: str


QUESTIONS = (
    Question("interests", "What area of technology interests you most right now?"),
    Question("experience", "How would you describe your current technical experience?"),
    Question("goals", "What would you like to achieve by the end of the program?"),
    Question("activities", "Which activities sound most appealing: building, analysing, designing, organising, or helping people?"),
    Question("work_type", "What kind of work would you prefer: independent, collaborative, structured, or customer-facing?"),
    Question("learning_preferences", "How do you prefer to learn, and how much time can you give each week?"),
)

START_PATTERNS = (
    r"which (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths)\b",
    r"what (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths) (?:should|would|do i|is|are|to)\b",
    r"best (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths) for me",
    r"what should i learn",
    r"(?:don['’]t|do not) know what to learn",
    r"(?:don['’]t|do not) know which (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths)\b",
    r"(?:don['’]t|do not) know (?:the|which) (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths) to register",
    r"not sure (?:which|what) (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths)\b",
    r"(?:which|what) (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths) should i register for",
    r"want to enter tech",
    r"help me choose",
    r"find the right (?:program|programme|track|path)",
    r"help me choose (?:a|the) (?:course|courses|program|programs|programme|programmes|track|tracks|path|paths) to register",
    r"which direction should i take",
    r"what direction should i take",
    r"which path should i take",
    r"help me choose a direction",
)
EXIT_WORDS = {"exit", "quit", "stop", "cancel", "normal chat", "leave"}
RESTART_WORDS = {
    "restart", "start over", "begin again", "try again",
    "another program", "explore another program", "try another program",
    "different program", "see other programs", "another track",
}
NORMAL_PATTERNS = (
    r"\b(fee|cost|tuition|refund|payment|gmail|google classroom|certificate|support|register|registration)\b",
    r"how long is techiestart",
    r"^\s*(who|what|when|where|why|how|can|could|do|does|did|is|are|will|would|which)\b",
    r"^\s*(tell me about|i want to know|i need to know|please explain)\b",
)


@dataclass(frozen=True)
class Program:
    program_id: str
    name: str
    description: str
    suitable_for: tuple[str, ...]
    prerequisites: tuple[str, ...]
    skills: tuple[str, ...]
    learning_outcomes: tuple[str, ...]
    career_directions: tuple[str, ...]
    duration: str


@dataclass(frozen=True)
class Match:
    program_id: str
    program_name: str
    score: int
    strength: str
    factors: tuple[str, ...]


def _tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(str(item) for item in value)
    return ()


def load_program_catalog() -> tuple[Program, ...]:
    configured_path = os.getenv("PATHFINDER_PROGRAM_CATALOG")
    catalog_path = Path(configured_path) if configured_path else Path(__file__).resolve().parents[2] / "data" / "programs.json"
    if configured_path and not catalog_path.is_absolute():
        catalog_path = Path.cwd() / catalog_path
    if not catalog_path.exists():
        return ()
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()

    programs = payload.get("programs", []) if isinstance(payload, dict) else []
    return tuple(
        Program(
            program_id=str(item["program_id"]),
            name=str(item["name"]),
            description=str(item.get("description", "")),
            suitable_for=_tuple(item.get("suitable_for")),
            prerequisites=_tuple(item.get("prerequisites")),
            skills=_tuple(item.get("skills")),
            learning_outcomes=_tuple(item.get("learning_outcomes")),
            career_directions=_tuple(item.get("career_directions")),
            duration=str(item.get("duration", "")),
        )
        for item in programs
        if (
            isinstance(item, dict)
            and item.get("program_id")
            and item.get("name")
            and not str(item["program_id"]).startswith("REPLACE_WITH_")
            and not str(item["name"]).startswith("REPLACE_WITH_")
        )
    )


def _words(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def match_programs(profile: dict[str, str], programs: tuple[Program, ...]) -> tuple[Match, ...]:
    profile_words = _words(" ".join(profile.values()))
    matches = []
    for program in programs:
        fields = {
            "interests": program.suitable_for + program.skills + program.career_directions,
            "experience": program.prerequisites,
            "goals": program.learning_outcomes + program.career_directions,
            "activities": program.skills + program.learning_outcomes,
            "work_type": program.suitable_for,
            "learning_preferences": program.suitable_for,
        }
        factors = []
        score = 0
        for key, values in fields.items():
            overlap = profile_words & _words(" ".join(values))
            if overlap:
                score += min(2, len(overlap))
                factors.append(f"Your {key.replace('_', ' ')} aligns with {program.name}.")
        strength = "strong" if score >= 6 else "possible" if score >= 2 else "unclear"
        matches.append(Match(program.program_id, program.name, score, strength, tuple(factors)))
    return tuple(sorted(matches, key=lambda item: item.score, reverse=True))


def is_pathfinder_request(message: str) -> bool:
    lowered = message.lower().strip()
    return any(re.search(pattern, lowered) for pattern in START_PATTERNS)


def wants_new_pathfinder(message: str) -> bool:
    """True for anything that should (re)start the Pathfinder from question 1:
    an explicit request (is_pathfinder_request) OR a restart phrase. Checked
    unconditionally by the router regardless of whether a session already
    exists, so this also covers restarting after a session has completed or
    been abandoned — not just while one is active."""
    lowered = message.lower().strip()
    return is_pathfinder_request(message) or lowered in RESTART_WORDS


def is_normal_question(message: str) -> bool:
    lowered = message.lower()
    return any(re.search(pattern, lowered) for pattern in NORMAL_PATTERNS)


def _get_session(db: Session, session_id: str) -> PathfinderSession | None:
    return db.query(PathfinderSession).filter(PathfinderSession.session_id == session_id).first()


def completed_match(session: PathfinderSession) -> tuple[dict[str, str], dict[str, object]] | None:
    """Returns (profile, result) for a just-completed session with a real
    match, or None if the session isn't completed or no catalog match was
    found (e.g. no program catalog configured yet). Used by the router to
    decide whether to layer a Gemini/RAG explanation on top of the
    deterministic recommendation."""
    if session.status != "completed":
        return None
    state = _state(session)
    result = state.get("result")
    profile = state.get("profile")
    if not isinstance(result, dict) or not isinstance(profile, dict):
        return None
    return profile, result


def _state(session: PathfinderSession) -> dict[str, object]:
    try:
        value = json.loads(session.state_json or "{}")
    except json.JSONDecodeError:
        value = {}
    return value if isinstance(value, dict) else {}


def _save_state(session: PathfinderSession, state: dict[str, object]) -> None:
    session.state_json = json.dumps(state)
    session.updated_at = datetime.utcnow()


def start_pathfinder(db: Session, session_id: str) -> str:
    session = _get_session(db, session_id)
    if session is None:
        session = PathfinderSession(session_id=session_id)
        db.add(session)
    session.status = "active"
    session.current_question = 0
    _save_state(session, {"answers": {}, "profile": {}, "result": None})
    db.commit()
    return "I can help you explore the available TechieStart programs. You can answer in your own words, and you can say 'exit' at any time.\n\n" + QUESTIONS[0].prompt


def process_pathfinder_message(db: Session, session_id: str, message: str) -> str:
    session = _get_session(db, session_id)
    if session is None or session.status != "active":
        return start_pathfinder(db, session_id)

    lowered = message.lower().strip()
    if lowered in EXIT_WORDS:
        session.status = "abandoned"
        db.commit()
        return "No problem. We can continue with normal TechieStart questions whenever you are ready."
    if lowered in RESTART_WORDS:
        return start_pathfinder(db, session_id)

    state = _state(session)
    answers = state.setdefault("answers", {})
    if not isinstance(answers, dict):
        answers = {}
        state["answers"] = answers
    question = QUESTIONS[session.current_question]
    answers[question.key] = message.strip()

    if session.current_question < len(QUESTIONS) - 1:
        session.current_question += 1
        _save_state(session, state)
        db.commit()
        return QUESTIONS[session.current_question].prompt

    profile = {str(key): str(value) for key, value in answers.items()}
    matches = match_programs(profile, load_program_catalog())
    state["profile"] = profile
    state["result"] = asdict(matches[0]) if matches else None
    session.status = "completed"
    _save_state(session, state)
    db.commit()

    if not matches:
        return (
            "Thanks. I have your Pathfinder profile, but the project does not yet have an "
            "authoritative program catalog with track names and requirements. I won't guess "
            "a recommendation. Please contact TechieStart support for the current track options."
        )
    best = matches[0]
    factor_text = " ".join(best.factors[:3]) or "Your answers provide a possible fit."
    return (
        f"Based on your answers, the strongest match is **{best.program_name}** "
        f"({best.strength} match). {factor_text}\n\n"
        "Would you like to view its details, ask a question, explore another program, or register?"
    )
