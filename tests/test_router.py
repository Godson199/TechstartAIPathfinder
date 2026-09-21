import json
import unittest
from unittest.mock import patch

import app.bot.router as router


class FakeChunk:
    def __init__(self, source, section, content, embedding):
        self.source = source
        self.section = section
        self.content = content
        self.embedding = embedding


class FakeQuery:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class FakeDB:
    def __init__(self, rows):
        self._rows = rows

    def query(self, model):
        return FakeQuery(self._rows)


class RetrieveContextTests(unittest.TestCase):
    def test_prefers_embedding_similarity_when_terms_do_not_overlap(self):
        query = "What documents are needed to apply?"
        training_chunk = FakeChunk(
            source="faq",
            section="Training track",
            content="The training track helps you build practical skills.",
            embedding=json.dumps([0.0, 1.0]),
        )
        apply_chunk = FakeChunk(
            source="faq",
            section="Application steps",
            content="Please submit your CV and LinkedIn profile.",
            embedding=json.dumps([1.0, 0.0]),
        )

        db = FakeDB([training_chunk, apply_chunk])

        with patch.object(router, "embed", return_value=[1.0, 0.0]):
            result = router.retrieve_context(db, query)

        self.assertIn(apply_chunk.content, result)
        self.assertLess(result.index(apply_chunk.content), result.index(training_chunk.content))

    def test_program_list_question_uses_faq_track_names_when_available(self):
        faq_context = """
        TechieStart offers the following listed tracks:
        1. Gen AI Content Creation
        2. Front-End Web Development
        3. AI & ML
        4. Data Analysis
        """

        reply = router._build_program_list_reply("What programs do you offer?", faq_context)

        self.assertIn("Gen AI Content Creation", reply)
        self.assertIn("Front-End Web Development", reply)
        self.assertIn("Data Analysis", reply)
        self.assertNotIn("I am not sure about the full list of programs offered", reply)


if __name__ == "__main__":
    unittest.main()
