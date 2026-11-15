import os
import unittest
from unittest.mock import patch

from src.ltm.llm import GeminiError, generate


class GeminiGenerationTests(unittest.TestCase):
    def test_generation_uses_requested_output_token_budget(self):
        response = {
            "candidates": [{
                "finishReason": "STOP",
                "content": {"parts": [{"text": "A complete draft"}]},
            }]
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("src.ltm.llm.setting", return_value="gemini-test"):
                with patch("src.ltm.llm._post", return_value=response) as post:
                    self.assertEqual(generate("Write a draft", max_output_tokens=3000), "A complete draft")
        payload = post.call_args.args[1]
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 3000)

    def test_max_tokens_finish_reason_is_reported_as_incomplete(self):
        response = {
            "candidates": [{
                "finishReason": "MAX_TOKENS",
                "content": {"parts": [{"text": "Partial draft"}]},
            }]
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch("src.ltm.llm.setting", return_value="gemini-test"):
                with patch("src.ltm.llm._post", return_value=response):
                    with self.assertRaisesRegex(GeminiError, "stopped at the output token limit"):
                        generate("Write a long draft", max_output_tokens=100)


if __name__ == "__main__":
    unittest.main()
