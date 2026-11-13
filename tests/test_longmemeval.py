import json
import tempfile
import unittest
from pathlib import Path

from src.ltm.longmemeval import run_longmemeval


class LongMemEvalAdapterTests(unittest.TestCase):
    def test_duplicate_session_ids_do_not_inflate_metrics(self):
        from src.ltm.longmemeval import _rank_metrics

        scores = _rank_metrics(["s1", "s1", "s2"], ["s1"], 2)
        self.assertEqual(scores["precision_at_k"], 0.5)
        self.assertEqual(scores["recall_at_k"], 1.0)
        self.assertEqual(scores["reciprocal_rank"], 1.0)

    def test_session_retrieval_metrics_and_abstention_count(self):
        dataset = [
            {
                "question_id": "q1",
                "question_type": "single-session-user",
                "question": "Which framework is used for vision training?",
                "question_date": "2024/01/01",
                "haystack_session_ids": ["s1", "s2"],
                "haystack_dates": ["2023/12/01", "2023/12/02"],
                "haystack_sessions": [
                    [{"role": "user", "content": "I use PyTorch for vision training."}],
                    [{"role": "user", "content": "I enjoy baking bread."}],
                ],
                "answer_session_ids": ["s1"],
            },
            {
                "question_id": "q2_abs",
                "question_type": "multi-session",
                "question": "What is the user's secret spaceship?",
                "question_date": "2024/01/01",
                "haystack_session_ids": ["s3"],
                "haystack_dates": ["2023/12/03"],
                "haystack_sessions": [
                    [{"role": "user", "content": "I like walking outdoors."}],
                ],
                "answer_session_ids": [],
            },
        ]
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "fixture.json"
            path.write_text(json.dumps(dataset), encoding="utf-8")
            report = run_longmemeval(path, k=1)

        self.assertEqual(report["questions_total"], 2)
        self.assertEqual(report["questions_scored"], 1)
        self.assertEqual(report["no_gold_evidence_questions"], 1)
        self.assertEqual(report["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(report["results"][0]["retrieval_results"][0]["session_id"], "s1")


if __name__ == "__main__":
    unittest.main()
