import unittest
from unittest.mock import Mock

from src.ltm.content import (
    CONTENT_TYPES, build_content_prompt, create_content, editorial_review,
    revise_content,
)


class ContentCreationTests(unittest.TestCase):
    def test_supported_content_formats(self):
        self.assertEqual(
            set(CONTENT_TYPES),
            {"blog", "article", "social", "email", "newsletter", "report", "script", "product"},
        )

    def test_prompt_includes_brief_and_relevant_memory(self):
        prompt = build_content_prompt(
            "Responsible AI", "article", "university students", "friendly",
            "long", ["privacy", "fairness"], "Include examples.",
            [{"content": "The project focuses on persistent assistant memory."}],
        )
        for expected in ("Responsible AI", "university students", "friendly",
                         "privacy, fairness", "Include examples.",
                         "persistent assistant memory"):
            self.assertIn(expected, prompt)

    def test_create_content_retrieves_memory_and_returns_trace(self):
        store = object()
        retriever = Mock(return_value=[{"id": 17, "content": "Audience prefers practical examples"}])
        generator = Mock(return_value="# A useful article\n\nDraft text.")
        result = create_content(
            store, "alice", "AI memory", content_type="article",
            retriever=retriever, generator=generator,
        )
        retriever.assert_called_once()
        generator.assert_called_once()
        self.assertEqual(result["memory_ids"], [17])
        self.assertEqual(result["memory_count"], 1)
        self.assertIn("A useful article", result["content"])

    def test_invalid_type_and_empty_topic_rejected(self):
        with self.assertRaises(ValueError):
            build_content_prompt("topic", content_type="unknown")
        with self.assertRaises(ValueError):
            build_content_prompt(" ")

    def test_revision_uses_instructions(self):
        generator = Mock(return_value="Revised draft")
        result = revise_content("Original draft", "Make it shorter", generator=generator)
        self.assertEqual(result, "Revised draft")
        self.assertIn("Make it shorter", generator.call_args.args[0])

    def test_editorial_review_parses_json(self):
        generator = Mock(return_value='{"clarity":"good","improvements":["Add an example"]}')
        result = editorial_review("A draft", generator=generator)
        self.assertEqual(result["clarity"], "good")
        self.assertEqual(result["improvements"], ["Add an example"])


if __name__ == "__main__":
    unittest.main()
