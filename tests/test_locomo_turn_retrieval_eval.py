import unittest

from scripts.build_locomo_turn_retrieval_eval import (
    build_locomo_cases,
    paired_summary,
)
from scripts.build_longmemeval_turn_retrieval_eval import (
    BM25_ABLATION_STOPWORDS,
    evaluate_turn_retrieval,
    rank_bm25,
)


class LoCoMoRetrievalTests(unittest.TestCase):
    def fixture(self):
        return [{
            "sample_id": "conv-test",
            "conversation": {
                "speaker_a": "Alex",
                "speaker_b": "Sam",
                "session_1": [
                    {"speaker": "Alex", "dia_id": "D1:1", "text": "I enjoy hiking."},
                    {"speaker": "Sam", "dia_id": "D1:2", "text": "I moved to Oslo last spring."},
                ],
                "session_1_date_time": "2023-01-01",
                "session_2": [
                    {"speaker": "Alex", "dia_id": "D2:1", "text": "The weather is sunny."},
                ],
                "session_2_date_time": "2023-02-01",
            },
            "qa": [
                {"question": "Which city did Sam move to?", "answer": "Oslo",
                 "evidence": ["D1:2"], "category": 1},
                {"question": "What happened in an unknown session?", "answer": "unknown",
                 "evidence": ["D9:9"], "category": 3},
                {"question": "Was Sam ever a pilot?", "adversarial_answer": "No",
                 "evidence": [], "category": 5},
                {"question": "When was the weather sunny?", "answer": "February",
                 "evidence": ["D2:1"], "category": 2},
            ],
        }]

    def test_builds_session_turns_and_maps_evidence_ids(self):
        cases, summary = build_locomo_cases(self.fixture())
        self.assertEqual(len(cases), 2)
        self.assertEqual(summary["eligible_questions"], 2)
        self.assertEqual(summary["excluded_by_reason"]["evidence_id_not_found"], 1)
        self.assertEqual(summary["excluded_by_reason"]["abstention_or_adversarial"], 1)
        case = cases[0]
        self.assertEqual(case["question_type"], "single-hop")
        gold = [turn for turn in case["turns"] if turn["relevant"]]
        self.assertEqual([turn["turn_id"] for turn in gold], ["D1:2"])
        self.assertEqual(gold[0]["session_id"], "session_1")
        self.assertEqual(gold[0]["date"], "2023-01-01")

    def test_frozen_stopword_ablation_is_callable_on_locomo_cases(self):
        cases, _ = build_locomo_cases(self.fixture())
        result = evaluate_turn_retrieval(
            cases,
            lambda question, turns: rank_bm25(
                question, turns, stopwords=BM25_ABLATION_STOPWORDS
            ),
            ks=(1, 3, 5, 10),
        )
        self.assertEqual(result["metrics"]["questions"], 2)
        self.assertGreaterEqual(result["metrics"]["recall_at_5"], 0.0)
        self.assertLessEqual(result["metrics"]["recall_at_5"], 1.0)

    def test_paired_summary_reports_question_level_changes(self):
        cases, _ = build_locomo_cases(self.fixture())
        baseline = evaluate_turn_retrieval(cases, rank_bm25, ks=(1, 3, 5, 10))
        ablation = evaluate_turn_retrieval(
            cases,
            lambda question, turns: rank_bm25(
                question, turns, stopwords=BM25_ABLATION_STOPWORDS
            ),
            ks=(1, 3, 5, 10),
        )
        paired = paired_summary(baseline, ablation, resamples=100)
        self.assertEqual(paired["matched_questions"], 2)
        self.assertIn("improved", paired["metrics"]["recall_at_5"])
        self.assertEqual(len(paired["metrics"]["recall_at_5"]["conversation_cluster_bootstrap_95_ci"]), 2)
        self.assertEqual(paired["conversation_clusters"], 1)


if __name__ == "__main__":
    unittest.main()
