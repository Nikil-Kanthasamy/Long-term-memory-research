import importlib.util
import unittest

try:
    import torch as _torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class TinyLanguageModelTests(unittest.TestCase):
    def test_byte_tokenizer_round_trip(self):
        from src.ltm.tiny_lm import ByteTokenizer
        sample = "Memory works: café 🧠"
        self.assertEqual(ByteTokenizer.decode(ByteTokenizer.encode(sample)), sample)


    def test_memory_prompt_keeps_evidence_and_question_within_context(self):
        from src.ltm.tiny_lm import ByteTokenizer, build_memory_prompt
        memory = "SQLite stores long-term memories."
        question = "Where are persistent memories saved?"
        prompt = build_memory_prompt(question, [memory], 128, include_memory=True)
        self.assertLessEqual(len(ByteTokenizer.encode(prompt)), 96)
        self.assertIn(memory, prompt)
        self.assertIn(question, prompt)

    def test_memory_prompt_without_memory_is_compact(self):
        from src.ltm.tiny_lm import ByteTokenizer, build_memory_prompt
        question = "What is the training objective?"
        prompt = build_memory_prompt(question, [], 128, include_memory=False)
        self.assertLessEqual(len(ByteTokenizer.encode(prompt)), 128)
        self.assertIn(question, prompt)
        self.assertNotIn("M:", prompt)

    def test_transformer_forward_loss_and_generation(self):
        import torch
        from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig

        config = TinyLMConfig(block_size=16, d_model=32, n_heads=4, n_layers=2, dropout=0.0)
        model = TinyLanguageModel(config)
        inputs = torch.randint(0, 256, (2, 12))
        targets = torch.randint(0, 256, (2, 12))
        logits, loss = model(inputs, targets)
        self.assertEqual(tuple(logits.shape), (2, 12, 256))
        self.assertTrue(torch.isfinite(loss).item())
        generated = model.generate(inputs[:1, :4], max_new_tokens=3, temperature=1.0, top_k=10)
        self.assertEqual(tuple(generated.shape), (1, 7))


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is not installed")
class TinyLMSupervisedTuningTests(unittest.TestCase):
    def test_prompt_labels_are_masked_and_answer_labels_remain(self):
        from scripts.train_tiny_lm_sft import encode_example
        prompt = "Q:"
        response = "A"
        inputs, labels = encode_example(prompt, response, 32)
        self.assertEqual(len(inputs), len(labels))
        self.assertEqual(labels[:len(prompt.encode()) - 1], [-100])
        self.assertIn(ord("A"), labels)

    def test_batch_padding_is_ignored(self):
        import torch
        from scripts.train_tiny_lm_sft import make_batch
        rows = [
            {"prompt": "Q:1 A:", "response": "one"},
            {"prompt": "Q:2 A:", "response": "two longer"},
        ]
        x, labels = make_batch(rows, 32, "cpu")
        self.assertEqual(tuple(x.shape), tuple(labels.shape))
        self.assertTrue((labels == -100).any().item())

    def test_memory_qa_evaluator_reports_both_conditions(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.evaluate_tiny_lm_memory_qa import load_cases, normalize
        self.assertEqual(normalize("SQLite Database!"), "sqlite database")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps({"question": "Where?", "memories": ["SQLite"], "expected_answer": "SQLite"}) + "\n", encoding="utf-8")
            self.assertEqual(len(load_cases(path)), 1)

    def test_retrieval_evidence_hit_is_separate_from_generation(self):
        from scripts.evaluate_tiny_lm_memory_qa import retrieval_contains_expected
        retrieved = [{"content": "Long-term memories are stored in a SQLite database."}]
        self.assertTrue(retrieval_contains_expected(retrieved, "SQLite database"))
        self.assertFalse(retrieval_contains_expected(retrieved, "PostgreSQL"))
        self.assertFalse(retrieval_contains_expected(retrieved, ""))

    def test_expanded_instruction_and_heldout_sets_are_distinct(self):
        from scripts.train_tiny_lm_sft import load_examples
        from scripts.evaluate_tiny_lm_memory_qa import load_cases
        training = load_examples("examples/tiny_lm_sft_demo.jsonl")
        heldout = load_cases("examples/tiny_lm_memory_qa_eval.jsonl")
        self.assertGreaterEqual(len(training), 40)
        self.assertGreaterEqual(len({row["prompt"] for row in training}), 40)
        self.assertGreaterEqual(len(heldout), 12)
        self.assertEqual(len({row["id"] for row in heldout}), len(heldout))


if __name__ == "__main__":
    unittest.main()
