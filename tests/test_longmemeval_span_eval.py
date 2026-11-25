import unittest

from scripts.build_longmemeval_span_eval import build_cases


class LongMemEvalSpanBuilderTests(unittest.TestCase):
    def test_uses_only_labeled_turns_and_keeps_exact_answer(self):
        data = [{
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "Which city was mentioned?",
            "answer": "Oslo",
            "haystack_sessions": [[
                {"role": "user", "content": "I visited Paris.", "has_answer": False},
                {"role": "assistant", "content": "Your trip to Oslo sounded lovely.", "has_answer": True},
            ]],
        }]
        rows, summary = build_cases(data, block_size=128)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["memories"], ["Your trip to Oslo sounded lovely."])
        self.assertEqual(rows[0]["expected_answer"], "Oslo")
        self.assertEqual(summary["eligible_questions"], 1)

    def test_skips_abstention_and_nonliteral_answers(self):
        data = [
            {
                "question_id": "q_abs",
                "question_type": "abstention",
                "question": "What is the secret?",
                "answer": "unknown",
                "haystack_sessions": [],
            },
            {
                "question_id": "q2",
                "question_type": "multi-session",
                "question": "What is the answer?",
                "answer": "New York",
                "haystack_sessions": [[
                    {"role": "user", "content": "I moved to NYC.", "has_answer": True},
                ]],
            },
        ]
        rows, summary = build_cases(data, block_size=128)
        self.assertEqual(rows, [])
        self.assertEqual(summary["skipped_by_reason"]["abstention"], 1)
        self.assertEqual(summary["skipped_by_reason"]["answer_not_literal_in_retained_evidence"], 1)

    def test_skips_questions_that_do_not_fit_context(self):
        data = [{
            "question_id": "q3",
            "question_type": "single-session-user",
            "question": "x" * 200,
            "answer": "Oslo",
            "haystack_sessions": [[
                {"role": "user", "content": "Oslo", "has_answer": True},
            ]],
        }]
        rows, summary = build_cases(data, block_size=128)
        self.assertEqual(rows, [])
        self.assertEqual(summary["skipped_by_reason"]["question_exceeds_context"], 1)


if __name__ == "__main__":
    unittest.main()
