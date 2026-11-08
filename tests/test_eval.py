import tempfile
import unittest
from pathlib import Path

from src.ltm.eval import evaluate_cases, score_ranking
from src.ltm.store import MemoryStore


class RetrievalEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(str(Path(self.tmp.name) / "eval.sqlite3"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_ranking_metrics(self):
        scores = score_ranking([8, 4, 2], [4, 2], k=3)
        self.assertAlmostEqual(scores["precision_at_k"], 2 / 3)
        self.assertEqual(scores["recall_at_k"], 1.0)
        self.assertEqual(scores["reciprocal_rank"], 0.5)

    def test_lexical_evaluation(self):
        relevant = self.store.add_memory("alice", "Uses PyTorch for vision training")
        self.store.add_memory("alice", "Prefers concise written explanations")
        report = evaluate_cases(
            self.store,
            [{"query": "PyTorch vision", "relevant_memory_ids": [relevant]}],
            user_id="alice",
            mode="lexical",
            k=1,
        )
        self.assertEqual(report["cases"], 1)
        self.assertEqual(report["metrics"]["recall_at_k"], 1.0)
        self.assertEqual(report["results"][0]["retrieved_ids"], [relevant])

    def test_importance_does_not_rescue_zero_overlap(self):
        self.store.add_memory("alice", "A completely unrelated gardening fact", importance=1.0)
        self.assertEqual(self.store.search("alice", "quantum computing", mode="lexical"), [])

    def test_semantic_mode_requires_embedding(self):
        self.store.add_memory("alice", "Uses PyTorch")
        with self.assertRaises(ValueError):
            self.store.search("alice", "framework", mode="semantic")

    def test_empty_relevant_set_is_abstention_safe(self):
        scores = score_ranking([1, 2], [], k=2)
        self.assertEqual(scores["recall_at_k"], 1.0)
        self.assertEqual(scores["reciprocal_rank"], 0.0)


if __name__ == "__main__":
    unittest.main()
