import unittest

from scripts.build_longmemeval_retrieval_eval import build_retrieval_cases, evaluate, rank_sessions, rank_sessions_bm25


class LongMemEvalRetrievalTests(unittest.TestCase):
    def test_builds_session_documents_and_marks_all_evidence(self):
        data = [{
            "question_id": "q1", "question_type": "multi-session",
            "question": "Where did I move?",
            "answer_session_ids": ["s2", "s3"],
            "haystack_session_ids": ["s1", "s2", "s3"],
            "haystack_sessions": [
                [{"role": "user", "content": "I like tea."}],
                [{"role": "user", "content": "I moved to Oslo."}],
                [{"role": "assistant", "content": "Your new home is Oslo."}],
            ],
        }]
        cases, summary = build_retrieval_cases(data)
        self.assertEqual(summary["eligible_questions"], 1)
        self.assertEqual([x["session_id"] for x in cases[0]["sessions"] if x["relevant"]], ["s2", "s3"])

    def test_skips_missing_or_unmatched_evidence(self):
        data = [{
            "question_id": "q", "question": "Where?", "answer_session_ids": ["missing"],
            "haystack_session_ids": ["s1"], "haystack_sessions": [[{"content": "Here."}]],
        }]
        cases, summary = build_retrieval_cases(data)
        self.assertEqual(cases, [])
        self.assertEqual(summary["skipped_by_reason"]["evidence_not_in_haystack"], 1)

    def test_retrieval_metrics_and_rank(self):
        sessions = [
            {"session_id": "a", "text": "I enjoy tea and coffee", "relevant": False},
            {"session_id": "b", "text": "I moved to Oslo last summer", "relevant": True},
        ]
        ranked = rank_sessions("Where did I move to Oslo?", sessions)
        self.assertEqual(ranked[0][1], "b")
        metrics = evaluate([{"question": "Where did I move to Oslo?", "sessions": sessions, "question_type": "multi-session"}])
        self.assertEqual(metrics["recall_at_1"], 1.0)
        self.assertEqual(metrics["mrr"], 1.0)
        bm25 = rank_sessions_bm25("Where did I move to Oslo?", sessions)
        self.assertEqual(bm25[0][1], "b")


if __name__ == "__main__":
    unittest.main()
