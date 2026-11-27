import unittest

from scripts.build_longmemeval_turn_retrieval_eval import (
    BM25_ABLATION_STOPWORDS, BM25_SELECTIVE_STOPWORDS, build_failure_analysis, build_turn_cases, evaluate_turn_retrieval, rank_bm25, rank_jaccard,
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

    def test_failure_analysis_includes_rank_buckets_and_readable_gold_context(self):
        cases, _ = build_turn_cases(self.fixture())
        result = evaluate_turn_retrieval(cases, rank_bm25, ks=(1, 3, 5))
        analysis = build_failure_analysis(cases, result)
        self.assertEqual(analysis["rank_of_first_gold_turn"]["rank_1"], 1)
        self.assertEqual(analysis["rank_of_first_gold_turn"]["not_retrieved"], 0)
        example = analysis["worst_examples"][0]
        self.assertEqual(example["question_id"], "q1")
        self.assertIn("Oslo", example["gold_turns"][0]["text"])
        self.assertIn("text", example["top_ranked_turns"][0])

    def test_bm25_length_normalization_parameter_changes_ranking(self):
        turns = [
            {"turn_id": "0:0", "text": "apple " + "filler " * 100, "relevant": False},
            {"turn_id": "0:1", "text": "apple", "relevant": True},
        ]
        without_length_normalization = rank_bm25("apple", turns, b=0.0)
        with_length_normalization = rank_bm25("apple", turns, b=1.0)
        self.assertEqual(without_length_normalization[0][1], "0:0")
        self.assertEqual(with_length_normalization[0][1], "0:1")


    def test_stopword_ablation_is_explicit_and_does_not_change_default_bm25(self):
        turns = [
            {"turn_id": "0:0", "text": "what was the location of the museum", "relevant": False},
            {"turn_id": "0:1", "text": "museum", "relevant": True},
        ]
        baseline = rank_bm25("what was the museum", turns)
        explicit_default = rank_bm25("what was the museum", turns, stopwords=frozenset())
        ablated = rank_bm25(
            "what was the museum", turns, stopwords=BM25_ABLATION_STOPWORDS
        )
        self.assertEqual(baseline, explicit_default)
        self.assertIn("what", BM25_ABLATION_STOPWORDS)
        self.assertNotIn("museum", BM25_ABLATION_STOPWORDS)
        self.assertEqual(len(ablated), 2)

        selective = rank_bm25(
            "what did I move to Oslo?",
            [{"turn_id": "0:0", "text": "I moved to Oslo in spring.", "relevant": True}],
            stopwords=BM25_SELECTIVE_STOPWORDS,
        )
        self.assertEqual(
            BM25_SELECTIVE_STOPWORDS, frozenset({"a", "an", "the", "and", "or"})
        )
        self.assertNotIn("what", BM25_SELECTIVE_STOPWORDS)
        self.assertNotIn("did", BM25_SELECTIVE_STOPWORDS)
        self.assertNotIn("to", BM25_SELECTIVE_STOPWORDS)
        self.assertEqual(len(selective), 1)

    def test_retrieval_ranks_gold_turn_and_reports_recall(self):
        cases, _ = build_turn_cases(self.fixture())
        for ranker in (rank_jaccard, rank_bm25):
            result = evaluate_turn_retrieval(cases, ranker, ks=(1, 3))
            self.assertEqual(result["metrics"]["recall_at_1"], 1.0)
            self.assertEqual(result["metrics"]["mrr"], 1.0)
            self.assertEqual(result["per_question"][0]["top_results"][0]["relevant"], True)


if __name__ == "__main__":
    unittest.main()
