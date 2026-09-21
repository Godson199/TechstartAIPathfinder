"""
One-time (and re-run-on-update) script to chunk, embed, and store the
Participant Handbook, curriculum, and FAQ content in Postgres for retrieval
by the chatbot.

Usage:
    1. Convert your source docs to markdown, e.g.:
         pandoc participant_handbook.docx -o data/participant_handbook.md
         pandoc curriculum.docx -o data/curriculum.md
       Write your FAQ as data/faq.md directly (Q/H2 headers work well).

    2. Run:
         python scripts/ingest_knowledge.py

Re-run any time the source docs change — this script clears and re-embeds
each source it's given, so it's safe to run repeatedly.
"""
import csv
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

from app.core.database import SessionLocal  # noqa: E402
from app.models.knowledge import KnowledgeChunk  # noqa: E402

client = OpenAI(
    api_key=os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
)
EMBEDDING_MODEL = "text-embedding-3-small"
MAX_WORDS_PER_CHUNK = 180  # keeps each chunk small enough for precise retrieval


def chunk_text(text: str, source: str, max_words: int = MAX_WORDS_PER_CHUNK) -> list[dict]:
    """Split markdown by headers (#, ##, ###) so each chunk stays on one topic,
    then further split long sections by word count."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        header_match = re.match(r"#{1,3} (.+)", sec)
        section_title = header_match.group(1).strip() if header_match else source
        words = sec.split()
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            if chunk_words:
                chunks.append({
                    "source": source,
                    "section": section_title,
                    "content": " ".join(chunk_words),
                })
    return chunks


def embed(text: str) -> list[float]:
    result = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return result.data[0].embedding


def load_csv_chunks(filepath: str, source_name: str) -> list[dict]:
    chunks = []
    with open(filepath, "r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            question = (row.get("Question") or "").strip()
            answer = (row.get("Answer") or "").strip()
            if not question or not answer:
                continue
            category = (row.get("Category") or source_name).strip()
            chunks.append({
                "source": source_name,
                "section": category,
                "content": f"Question: {question}\nAnswer: {answer}",
            })
    return chunks


def ingest(filepath: str, source_name: str, file_type: str = "markdown") -> None:
    if not os.path.exists(filepath):
        print(f"  skip: {filepath} not found")
        return

    if file_type == "csv":
        chunks = load_csv_chunks(filepath, source_name)
    else:
        with open(filepath, "r", encoding="utf-8") as file:
            text = file.read()
        chunks = chunk_text(text, source_name)

    db = SessionLocal()
    try:
        # Clear any previous chunks from this source before re-ingesting
        db.query(KnowledgeChunk).filter(KnowledgeChunk.source == source_name).delete()

        for c in chunks:
            vec = embed(c["content"])
            db.add(KnowledgeChunk(
                source=c["source"],
                section=c["section"],
                content=c["content"],
                embedding=json.dumps(vec),
            ))
        db.commit()
        print(f"  ingested {len(chunks)} chunks from {filepath}")
    finally:
        db.close()


if __name__ == "__main__":
    print("Ingesting TechieStart knowledge base...")
    ingest("data/participant_handbook.md", "handbook")
    ingest("data/curriculum.md", "curriculum")
    ingest("data/faq.md", "faq")
    ingest("data/TechieStart_50_FAQs_Guide.csv", "faq_csv", "csv")
    print("Done.")
