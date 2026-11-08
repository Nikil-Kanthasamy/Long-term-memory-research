import tempfile
import unittest
from pathlib import Path
from src.ltm.store import MemoryStore,cosine

class MemoryStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.store=MemoryStore(str(Path(self.tmp.name)/"test.sqlite3"))
    def tearDown(self):
        self.tmp.cleanup()
    def test_user_isolation(self):
        self.store.add_memory("alice","Uses PyTorch for model training")
        self.store.add_memory("bob","Uses TensorFlow for experiments")
        self.assertEqual(len(self.store.search("alice","PyTorch")),1)
        self.assertEqual(self.store.search("alice","TensorFlow"),[])
    def test_duplicate_guard(self):
        first=self.store.add_memory("alice","Prefers concise answers")
        second=self.store.add_memory("alice","Prefers concise answers")
        self.assertEqual(first,second)
        self.assertEqual(len(self.store.active_memories("alice")),1)
    def test_delete(self):
        mid=self.store.add_memory("alice","Temporary fact")
        self.assertTrue(self.store.delete_memory("alice",mid))
        self.assertFalse(self.store.active_memories("alice"))
    def test_cosine(self):
        self.assertAlmostEqual(cosine([1,0],[1,0]),1.0)
        self.assertAlmostEqual(cosine([1,0],[0,1]),0.0)

if __name__=="__main__":
    unittest.main()
