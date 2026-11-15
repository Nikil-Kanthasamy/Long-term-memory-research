import tempfile
import unittest
from pathlib import Path
from src.ltm.store import MemoryStore, cosine

class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(str(Path(self.tmp.name) / "test.sqlite3"))
    def tearDown(self):
        self.tmp.cleanup()
    def test_user_isolation(self):
        self.store.add_memory("alice", "Uses PyTorch for model training")
        self.store.add_memory("bob", "Uses TensorFlow for experiments")
        self.assertEqual(len(self.store.search("alice", "PyTorch")), 1)
        self.assertEqual(self.store.search("alice", "TensorFlow"), [])
    def test_duplicate_guard(self):
        a = self.store.add_memory("alice", "Prefers concise answers")
        b = self.store.add_memory("alice", "Prefers concise answers")
        self.assertEqual(a, b)
        self.assertEqual(len(self.store.active_memories("alice")), 1)
    def test_delete(self):
        mid = self.store.add_memory("alice", "Temporary fact")
        self.assertTrue(self.store.delete_memory("alice", mid))
        self.assertFalse(self.store.active_memories("alice"))
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1,0], [1,0]), 1.0)
        self.assertAlmostEqual(cosine([1,0], [0,1]), 0.0)
    def test_supersede_preserves_history(self):
        old = self.store.add_memory("alice", "Uses TensorFlow", category="project")
        new = self.store.add_memory("alice", "Uses PyTorch", category="project", supersedes_id=old)
        active = self.store.active_memories("alice")
        self.assertEqual([m["id"] for m in active], [new])
        self.assertEqual(active[0]["supersedes_id"], old)
        history = self.store.memory_history("alice", old)
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["status"], "superseded")
    def test_invalid_supersede_cannot_cross_users(self):
        old = self.store.add_memory("bob", "Private fact")
        with self.assertRaises(ValueError):
            self.store.add_memory("alice", "Replacement", supersedes_id=old)
    def test_temporal_filter(self):
        self.store.add_memory("alice", "Old project status", valid_from="2020-01-01T00:00:00+00:00",
                              valid_until="2021-01-01T00:00:00+00:00")
        self.assertEqual(self.store.search("alice", "project", as_of="2022-01-01T00:00:00+00:00"), [])
    def test_raw_turn_search_scoped(self):
        self.store.add_turn("alice", "The training uses PyTorch")
        self.store.add_turn("bob", "The training uses JAX")
        self.assertEqual(len(self.store.search_turns("alice", "PyTorch")), 1)
        self.assertEqual(self.store.search_turns("alice", "JAX"), [])

if __name__ == "__main__":
    unittest.main()
