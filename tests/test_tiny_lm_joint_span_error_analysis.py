import unittest

from scripts.analyze_tiny_lm_joint_span_errors import gold_span, summarize


class TinyLMJointSpanErrorAnalysisTests(unittest.TestCase):
    def test_gold_span_uses_byte_offsets_and_answer_independent_truncation(self):
        case = {
            "id": "one",
            "memories": ["prefix café answer suffix"],
            "question": "where?",
            "expected_answer": "answer",
        }
        start, end, width = gold_span(case, 128)
        self.assertEqual(start, len("prefix café ".encode("utf-8")))
        self.assertEqual(width, len("answer".encode("utf-8")))
        self.assertEqual(end - start + 1, width)

    def test_missing_answer_after_context_truncation_is_unscorable(self):
        case = {
            "id": "two",
            "memories": ["x" * 100],
            "question": "q" * 30,
            "expected_answer": "x" * 10,
        }
        self.assertIsNone(gold_span(case, 40))

    def test_summary_uses_joint_decoded_boundaries(self):
        rows = [
            {"exact": True, "start_error": 0, "end_error": 0,
             "abs_start_error": 0, "abs_end_error": 0, "length_error": 0},
            {"exact": False, "start_error": -3, "end_error": 0,
             "abs_start_error": 3, "abs_end_error": 0, "length_error": 3},
        ]
        result = summarize(rows)
        self.assertEqual(result["joint_start_boundary_accuracy"], 0.5)
        self.assertEqual(result["joint_end_boundary_accuracy"], 1.0)
        self.assertEqual(result["exact_span_accuracy"], 0.5)


if __name__ == "__main__":
    unittest.main()
