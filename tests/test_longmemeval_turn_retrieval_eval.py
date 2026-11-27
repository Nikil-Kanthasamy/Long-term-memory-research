import unittest

from scripts.build_longmemeval_turn_retrieval_eval import (
    build_turn_cases, evaluate_turn_retrieval, rank_bm25, rank_jaccard,
)


class LongMemEvalTurnRetrievalTests(unittest.TestCase):
    def fixture(self):
        return [{
            "question_id": "q1",
            "question_type": "multi-session",
            "question": "Which city did I move to?",
            "haystack_session_ids": ["s1", "s2"],
            "haystack_dates": ["2024/01/01", "2024/02/01"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "I enjoy drinking tea.", "has_answer": False},
                    {"role": "user", "content": "I moved to Oslo last spring.", "has_answer": True},
                ],
                [
                    {"role": "assistant", "content": "Your new apartment sounds nice.", "has_answer": False},
                ],
            ],
        }]

    def test_builds_turn_documents_and_uses_only_explicit_gold_labels(self):
        cases, summary = build_turn_cases(self.fixture())
        self.assertEqual(summary["eligible_questions"], 1)
        self.assertEqual(summary["gold_evidence_turns"], 1)
        self.assertEqual(cases[0]["turns"][1]["turn_id"], "0:1")
        self.assertTrue(cases[0]["turns"][1]["relevant"])
        self.assertFalse(cases[0]["turns"][0]["relevant"])
        self.assertEqual(cases[0]["turns"][1]["session_id"], "s1")

    def test_abstentions_and_unlabeled_questions_are_accounted_for(self):
        data = self.fixture() + [{
            "question_id": "q2_abs", "question": "Unknown?", "haystack_sessions": [],
            "haystack_session_ids": [], "answer_session_ids": [],
        }, {
            "question_id": "q3", "question": "What color?", "haystack_session_ids": ["s3"],
            "haystack_sessions": [[{"role": "user", "content": "The sky is blue.", "has_answer": False}]],
        }]
        cases, summary = build_turn_cases(data)
        self.assertEqual(len(cases), 1)
        self.assertEqual(summary["abstention_questions"], 1)
        self.assertEqual(summary["skipped_by_reason"]["no_labeled_evidence_turn"], 1)
        self.assertEqual(summary["answerable_questions"], 2)

    def test_retrieval_ranks_gold_turn_and_reports_recall(self):
        cases, _ = build_turn_cases(self.fixture())
        for ranker in (rank_jaccard, rank_bm25):
            result = evaluate_turn_retrieval(cases, ranker, ks=(1, 3))
            self.assertEqual(result["metrics"]["recall_at_1"], 1.0)
            self.assertEqual(result["metrics"]["mrr"], 1.0)
            self.assertEqual(result["per_question"][0]["top_results"][0]["relevant"], True)


if __name__ == "__main__":
    unittest.main()
