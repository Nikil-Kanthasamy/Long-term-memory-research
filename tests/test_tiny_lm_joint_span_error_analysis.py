import json
import tempfile
import unittest
from pathlib import Path

from scripts.analyze_tiny_lm_joint_span_errors import analyze, gold_span, summarize


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

    def test_signed_start_error_distinguishes_early_and_late(self):
        rows = [
            {"exact": False, "start_error": -2, "end_error": 0,
             "abs_start_error": 2, "abs_end_error": 0, "length_error": 2},
            {"exact": False, "start_error": 3, "end_error": 0,
             "abs_start_error": 3, "abs_end_error": 0, "length_error": -3},
        ]
        result = summarize(rows)
        self.assertEqual(result["fraction_start_too_early"], 0.5)
        self.assertEqual(result["fraction_start_too_late"], 0.5)
        self.assertEqual(result["fraction_predicted_span_too_long"], 0.5)
        self.assertEqual(result["fraction_predicted_span_too_short"], 0.5)


    def _analyze_predictions(self, predictions):
        cases = [{
            "id": "case-1",
            "memories": ["The code is ABC123."],
            "question": "What is the code?",
            "expected_answer": "ABC123",
        }]
        payload = {"after_joint_span_supervision": {"per_case": predictions}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seed.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            return analyze(cases, [str(path)], 128)

    def test_duplicate_case_predictions_are_rejected(self):
        row = {"case_id": "case-1", "predicted_start": 13, "predicted_end_inclusive": 18}
        with self.assertRaisesRegex(ValueError, "duplicate prediction"):
            self._analyze_predictions([row, row])

    def test_out_of_range_prediction_is_rejected(self):
        row = {"case_id": "case-1", "predicted_start": 0, "predicted_end_inclusive": 999}
        with self.assertRaisesRegex(ValueError, "outside retained memory"):
            self._analyze_predictions([row])

    def test_non_integer_offsets_are_rejected(self):
        row = {"case_id": "case-1", "predicted_start": "13", "predicted_end_inclusive": 18}
        with self.assertRaisesRegex(ValueError, "non-integer"):
            self._analyze_predictions([row])


if __name__ == "__main__":
    unittest.main()
