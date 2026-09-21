"""
End-to-end Pathfinder tests against the real FastAPI app + SQLite, covering
the scenarios called out in the implementation spec: complete beginners,
users interrupting with a normal question mid-assessment, exit/restart, and
a catalog-backed recommendation. The Gemini client is mocked throughout so
these run offline with no API key.
"""
import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_pathfinder_flow.db")
os.environ.setdefault("GEMINI_API_KEY", "test-key")
os.environ.setdefault("API_KEY", "test-key")

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
import app.bot.router as router  # noqa: E402

BEGINNER_ANSWERS = [
    "I like building things and web development",
    "complete beginner, never written code",
    "become a junior developer",
    "building things",
    "collaborative",
    "video lessons, about 5 hours a week",
]

SAMPLE_CATALOG = {
    "programs": [
        {
            "program_id": "dev",
            "name": "Software Development Track",
            "description": "Build web applications from scratch.",
            "suitable_for": ["beginners", "people who like building things", "collaborative"],
            "prerequisites": ["none"],
            "skills": ["building", "web development"],
            "learning_outcomes": ["build applications", "become a developer"],
            "career_directions": ["junior developer"],
            "duration": "3 months",
        },
        {
            "program_id": "design",
            "name": "Product Design Track",
            "description": "Design interfaces and user experiences.",
            "suitable_for": ["creative", "visual thinkers"],
            "prerequisites": ["none"],
            "skills": ["design", "prototyping"],
            "learning_outcomes": ["design interfaces"],
            "career_directions": ["product designer"],
            "duration": "3 months",
        },
    ]
}


def _fake_gemini_response(text: str) -> MagicMock:
    response = MagicMock()
    response.text = text
    return response


class PathfinderFlowTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.client.__enter__()  # triggers the startup event (creates tables)
        self.addCleanup(self.client.__exit__, None, None, None)
        # Every test starts a fresh session id so runs don't interfere.
        import uuid
        self.session_id = f"test-{uuid.uuid4()}"

    def _chat(self, message: str) -> str:
        response = self.client.post(
            "/api/bot/chat", json={"session_id": self.session_id, "message": message}
        )
        self.assertEqual(response.status_code, 200)
        return response.json()["reply"]

    # -- Complete beginner with the Data Analytics curriculum integrated --
    @patch.object(router, "get_gemini_client")
    def test_beginner_pathfinder_recommends_data_analytics(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "Data Analytics is a strong fit for someone who enjoys analyzing data and dashboards."
        )
        reply = self._chat("Which course is best for me?")
        self.assertIn("technology interests you most", reply)

        for answer in BEGINNER_ANSWERS:
            reply = self._chat(answer)

        self.assertIn("Data Analytics", reply)

    # -- Catalog-backed recommendation, Gemini explanation succeeds -------
    @patch.object(router, "get_gemini_client")
    def test_catalog_backed_recommendation_uses_gemini_explanation(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "The Software Development Track looks like a strong fit given your interest "
            "in building things. Want to see the curriculum, ask a question, explore "
            "another program, or register?"
        )
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = os.path.join(tmp, "programs.json")
            with open(catalog_path, "w") as f:
                json.dump(SAMPLE_CATALOG, f)
            with patch.dict(os.environ, {"PATHFINDER_PROGRAM_CATALOG": catalog_path}):
                self._chat("Which course is best for me?")
                for answer in BEGINNER_ANSWERS:
                    reply = self._chat(answer)

        self.assertIn("Software Development Track", reply)
        mock_client.return_value.models.generate_content.assert_called()

    # -- Gemini explanation fails -> falls back to the deterministic match
    @patch.object(router, "get_gemini_client")
    def test_catalog_backed_recommendation_falls_back_on_gemini_error(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("boom")
        with tempfile.TemporaryDirectory() as tmp:
            catalog_path = os.path.join(tmp, "programs.json")
            with open(catalog_path, "w") as f:
                json.dump(SAMPLE_CATALOG, f)
            with patch.dict(os.environ, {"PATHFINDER_PROGRAM_CATALOG": catalog_path}):
                self._chat("Which course is best for me?")
                for answer in BEGINNER_ANSWERS:
                    reply = self._chat(answer)

        self.assertIn("strongest match is", reply)
        self.assertIn("Software Development Track", reply)

    # -- Normal question mid-assessment doesn't lose the participant's place
    @patch.object(router, "get_gemini_client")
    def test_normal_question_mid_pathfinder_preserves_progress(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "The admin fee is \u20a610,000 and covers your seat for 3 months."
        )
        self._chat("I want to enter tech")  # question 1
        reply = self._chat("how much is the admin fee?")  # interruption
        self.assertIn("10,000", reply)

        reply = self._chat(BEGINNER_ANSWERS[0])  # should be treated as Q1's answer
        self.assertIn("technical experience", reply)  # i.e. now on question 2

    # -- Any natural RAG question mid-assessment preserves progress
    @patch.object(router, "get_gemini_client")
    def test_general_rag_question_mid_pathfinder_preserves_progress(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "You can apply using the registration process described by TechieStart support."
        )
        self._chat("I want to enter tech")  # question 1
        reply = self._chat("What documents are needed to apply")
        self.assertIn("registration process", reply)

        reply = self._chat(BEGINNER_ANSWERS[0])  # still answers question 1
        self.assertIn("technical experience", reply)

    # -- Exit, then normal chat resumes without re-entering the Pathfinder
    @patch.object(router, "get_gemini_client")
    def test_exit_then_normal_chat(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "Sure thing!"
        )
        self._chat("help me choose a program")
        reply = self._chat("exit")
        self.assertIn("normal TechieStart questions", reply)

        reply = self._chat("thanks")
        self.assertEqual(reply, "Sure thing!")

    # -- Restart after completion should start again from question 1
    @patch.object(router, "get_gemini_client")
    def test_restart_after_completion(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "Data Analytics is the strongest match for your profile."
        )
        self._chat("I want to enter tech")
        for answer in BEGINNER_ANSWERS:
            reply = self._chat(answer)
        self.assertIn("Data Analytics", reply)

        reply = self._chat("restart")
        self.assertIn("technology interests you most", reply)

    # -- "Explore another program" phrasing also restarts post-completion
    @patch.object(router, "get_gemini_client")
    def test_explore_another_program_restarts(self, mock_client):
        mock_client.return_value.models.generate_content.side_effect = RuntimeError("no catalog")
        self._chat("I want to enter tech")
        for answer in BEGINNER_ANSWERS:
            self._chat(answer)

        reply = self._chat("explore another program")
        self.assertIn("technology interests you most", reply)

    @patch.object(router, "get_gemini_client")
    def test_data_analytics_curriculum_question_uses_support_message(self, mock_client):
        mock_client.return_value.models.generate_content.return_value = _fake_gemini_response(
            "The Data Analytics curriculum includes Excel, SQL, Power BI, and Python."
        )

        response = self.client.post(
            "/api/bot/chat",
            json={
                "session_id": self.session_id,
                "message": "What are the specific curriculum details for the Data Analytics program?",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn("I am not sure about the specific curriculum details for the Data Analytics program.", response.json()["reply"])
        self.assertIn("TechieStart support line provided on the program page", response.json()["reply"])

    # -- Explicit button-triggered start
    def test_pathfinder_start_endpoint(self):
        response = self.client.post(
            "/api/bot/pathfinder/start", json={"session_id": self.session_id}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("technology interests you most", response.json()["reply"])


if __name__ == "__main__":
    unittest.main()
