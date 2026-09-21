import unittest

from app.bot.pathfinder import (
    Program,
    is_normal_question,
    is_pathfinder_request,
    match_programs,
)


class PathfinderTests(unittest.TestCase):
    def setUp(self):
        self.programs = (
            Program(
                program_id="data",
                name="Data Track",
                description="",
                suitable_for=("analysis", "numbers"),
                prerequisites=("beginner",),
                skills=("analysis", "spreadsheets"),
                learning_outcomes=("analyse data",),
                career_directions=("data analyst",),
                duration="3 months",
            ),
            Program(
                program_id="design",
                name="Design Track",
                description="",
                suitable_for=("creative", "visual"),
                prerequisites=("beginner",),
                skills=("design", "prototyping"),
                learning_outcomes=("design interfaces",),
                career_directions=("product designer",),
                duration="3 months",
            ),
        )

    def test_matches_profile_transparently(self):
        matches = match_programs(
            {
                "interests": "analysis and numbers",
                "experience": "beginner",
                "goals": "become a data analyst",
                "activities": "analysis",
            },
            self.programs,
        )

        self.assertEqual(matches[0].program_id, "data")
        self.assertGreater(matches[0].score, matches[1].score)
        self.assertTrue(matches[0].factors)

    def test_data_analytics_program_is_in_catalog(self):
        from app.bot.pathfinder import load_program_catalog

        programs = load_program_catalog()
        self.assertTrue(programs)
        names = {program.name.lower() for program in programs}
        self.assertIn("data analytics", names)

    def test_recognizes_pathfinder_and_normal_questions(self):
        self.assertTrue(is_pathfinder_request("Which course is best for me?"))
        self.assertTrue(is_pathfinder_request("Which program should I choose?"))
        self.assertTrue(is_pathfinder_request("What course should I choose?"))
        self.assertTrue(is_pathfinder_request("I don't know which programme to choose."))
        self.assertTrue(is_pathfinder_request("I don't know the programme to register for."))
        self.assertTrue(is_pathfinder_request("Which programme should I register for?"))
        self.assertTrue(is_pathfinder_request("Which direction should I take in tech?"))
        self.assertTrue(is_pathfinder_request("I'm not sure what path is right for me."))
        self.assertTrue(is_pathfinder_request("Which track is best for me?"))
        self.assertTrue(is_normal_question("How much is the admin fee?"))
        self.assertTrue(is_normal_question("What documents are needed to apply?"))
        self.assertTrue(is_normal_question("What documents are needed to apply"))
        self.assertTrue(is_normal_question("Tell me about the registration process"))
        self.assertTrue(is_normal_question("Can I use a Gmail address?"))
        self.assertFalse(is_normal_question("I enjoy building things"))


if __name__ == "__main__":
    unittest.main()