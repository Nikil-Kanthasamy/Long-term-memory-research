import importlib.util
import unittest

try:
    import torch as _torch  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@unittest.skipUnless(TORCH_AVAILABLE, "optional PyTorch dependency is not installed")
class TinyLanguageModelTests(unittest.TestCase):
    def test_byte_bpe_tokenizer_round_trip_and_compression(self):
        from src.ltm.bpe_tokenizer import ByteBPETokenizer
        sample = "memory memory memory stores persistent facts. café 🧠"
        tokenizer = ByteBPETokenizer.train(sample * 20, vocab_size=300)
        encoded = tokenizer.encode(sample)
        self.assertEqual(tokenizer.decode(encoded), sample)
        self.assertLess(len(encoded), len(sample.encode("utf-8")))
        restored = ByteBPETokenizer.from_dict(tokenizer.to_dict())
        self.assertEqual(restored.encode(sample), encoded)
        self.assertEqual(restored.vocab_size, tokenizer.vocab_size)

    def test_bpe_tokenizer_training_is_deterministic(self):
        from src.ltm.bpe_tokenizer import ByteBPETokenizer
        corpus = "the memory system stores facts\n" * 30
        first = ByteBPETokenizer.train(corpus, vocab_size=280)
        second = ByteBPETokenizer.train(corpus, vocab_size=280)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_bpe_memory_prompt_and_sft_preserve_context(self):
        from src.ltm.bpe_tokenizer import ByteBPETokenizer
        from src.ltm.tiny_lm import build_memory_prompt
        from scripts.train_tiny_lm_sft import encode_example
        memory = "SQLite stores persistent project memories."
        question = "Where are project memories stored?"
        tokenizer = ByteBPETokenizer.train((memory + " " + question) * 20, vocab_size=300)
        prompt = build_memory_prompt(question, [memory], 128, tokenizer=tokenizer)
        self.assertIn(memory, prompt)
        self.assertIn(question, prompt)
        inputs, labels = encode_example(
            f"M:{memory}\nQ:{question}\nA:", "SQLite.", 128, tokenizer
        )
        self.assertIn(question, tokenizer.decode(inputs))
        self.assertIn(memory, tokenizer.decode(inputs))
        self.assertTrue(any(label >= 0 for label in labels))

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

    def test_memory_sft_example_preserves_fact_and_question(self):
        from scripts.train_tiny_lm_sft import encode_example
        from src.ltm.tiny_lm import ByteTokenizer
        memory = "SQLite stores persistent memories."
        question = "Where are memories stored?"
        prompt = f"M:{memory}\\nQ:{question}\\nA:"
        inputs, labels = encode_example(prompt, "SQLite.", 128)
        decoded_prompt = ByteTokenizer.decode(inputs)
        self.assertIn(memory, decoded_prompt)
        self.assertIn(question, decoded_prompt)
        self.assertIn(ord("S"), labels)

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

    def test_evaluator_uses_shared_memory_prompt_builder(self):
        from scripts.evaluate_tiny_lm_memory_qa import build_condition_prompts
        from src.ltm.tiny_lm import TinyLMConfig
        question = "Where are persistent memories saved?"
        evidence = [{"content": "SQLite stores persistent memories."}]
        prompts = build_condition_prompts(question, evidence, 128)
        self.assertIn(question, prompts["memory"])
        self.assertIn("SQLite stores persistent memories.", prompts["memory"])
        self.assertIn(question, prompts["no_memory"])
        self.assertNotIn("SQLite stores", prompts["no_memory"])

    def test_evaluator_builds_irrelevant_memory_control(self):
        from scripts.evaluate_tiny_lm_memory_qa import build_condition_prompts
        question = "Which database stores the memories?"
        relevant = [{"content": "Long-term memories are stored in SQLite."}]
        distractor = ["The model uses a decoder-only Transformer."]
        prompts = build_condition_prompts(
            question, relevant, 128, distractor_memories=distractor
        )
        self.assertIn("SQLite", prompts["memory"])
        self.assertNotIn("SQLite", prompts["irrelevant_memory"])
        self.assertIn("decoder-only Transformer", prompts["irrelevant_memory"])
        self.assertIn(question, prompts["irrelevant_memory"])

    def test_retrieval_only_evidence_baseline(self):
        from scripts.evaluate_tiny_lm_memory_qa import retrieval_only_evidence
        from scripts.evaluate_tiny_lm_memory_qa import normalize
        retrieved = [
            {"content": "Long-term memories are stored in a SQLite database."},
            {"content": "The first prototype uses lexical memory retrieval."},
        ]
        evidence = retrieval_only_evidence(retrieved)
        self.assertIn("SQLite database", evidence)
        self.assertIn("lexical memory retrieval", evidence)
        self.assertIn("sqlite database", normalize(evidence))

    def test_teacher_forced_gold_answer_score_is_finite_and_token_counted(self):
        import torch
        from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig, ByteTokenizer
        from scripts.evaluate_tiny_lm_memory_qa import score_expected_response

        torch.manual_seed(7)
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=32, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0,
        ))
        result = score_expected_response(
            model, ByteTokenizer(), "M:SQLite\nQ:Where?\nA:",
            "SQLite", "cpu",
        )
        self.assertTrue(torch.isfinite(torch.tensor(result["mean_nll"])).item())
        self.assertGreater(result["perplexity"], 0)
        self.assertEqual(result["token_count"], len("SQLite\n".encode("utf-8")))

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
