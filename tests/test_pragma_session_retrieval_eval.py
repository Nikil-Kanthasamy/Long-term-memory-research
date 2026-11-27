import unittest

from scripts.build_pragma_session_retrieval_eval import (
    build_pragma_cases,
    paired_user_bootstrap,
)
from scripts.build_longmemeval_turn_retrieval_eval import (
    BM25_ABLATION_STOPWORDS,
    evaluate_turn_retrieval,
    rank_bm25,
)


class PragmaSessionRetrievalTests(unittest.TestCase):
    def histories(self):
        return [
            {
                "user_id": "user-a",
                "timestamps": [
                    "2025-05-01T00:00:00",
                    "2025-05-03T00:00:00",
                    "2025-05-06T00:00:00",
                ],
                "session_types": ["event", "filler", "trajectory"],
                "sessions": [
                    [
                        {"role": "user", "content": "I completed a beginner pottery course."},
                        {"role": "assistant", "content": "You enjoyed learning ceramics."},
                    ],
                    [{"role": "user", "content": "The weather is pleasant today."}],
                    [{"role": "user", "content": "I am considering an advanced pottery course."}],
                ],
            }
        ]

    def metadata(self):
        return [
            {
                "user_id": "user-a",
                "query_type": "type1",
                "query": "What activity should I build on from my recent experience?",
                "query_timestamp": "2025-05-04T00:00:00",
                "evidence_session_index": [0],
            },
            {
                "user_id": "user-a",
                "query_type": "type4",
                "query": "Should I abandon pottery despite my developing interest?",
                "query_timestamp": "2025-05-04T00:00:00",
                "evidence_session_index": [0],
            },
        ]

    def test_joins_sessions_and_excludes_future_history(self):
        cases, summary = build_pragma_cases(self.histories(), self.metadata())
        self.assertEqual(len(cases), 2)
        self.assertEqual(summary["users_in_history_file"], 1)
        self.assertEqual(summary["eligible_questions"], 2)
        self.assertEqual(len(cases[0]["turns"]), 2)
        self.assertEqual(cases[0]["gold_turn_count"], 1)
        self.assertEqual(
            [turn["turn_id"] for turn in cases[0]["turns"] if turn["relevant"]],
            ["user-a:0000"],
        )
        self.assertEqual(cases[0]["question_type"], "event-align")

    def test_rejects_gold_evidence_after_query_time(self):
        metadata = [{
            "user_id": "user-a",
            "query_type": "type3",
            "query": "What is my direction?",
            "query_timestamp": "2025-05-04T00:00:00",
            "evidence_session_index": [2],
        }]
        cases, summary = build_pragma_cases(self.histories(), metadata)
        self.assertEqual(cases, [])
        self.assertEqual(summary["excluded_by_reason"]["future_evidence"], 1)

    def test_frozen_ablation_and_user_cluster_bootstrap(self):
        cases, _ = build_pragma_cases(self.histories(), self.metadata())
        baseline = evaluate_turn_retrieval(cases, rank_bm25)
        ablation = evaluate_turn_retrieval(
            cases,
            lambda query, sessions: rank_bm25(
                query, sessions, stopwords=BM25_ABLATION_STOPWORDS
            ),
        )
        paired = paired_user_bootstrap(baseline, ablation, resamples=100)
        self.assertEqual(paired["matched_questions"], 2)
        self.assertEqual(paired["user_clusters"], 1)
        self.assertIn(
            "user_cluster_bootstrap_95_ci",
            paired["metrics"]["recall_at_5"],
        )


if __name__ == "__main__":
    unittest.main()
