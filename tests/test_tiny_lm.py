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

    def test_generation_accepts_multi_token_stop_sequence(self):
        import torch
        from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=16, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0,
        ))
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        from unittest.mock import patch
        with patch("torch.multinomial", side_effect=[
            torch.tensor([[255]]), torch.tensor([[254]]),
        ]):
            generated = model.generate(
                prompt, max_new_tokens=4, temperature=1.0, top_k=1,
                stop_sequence_ids=[255, 254],
            )
        self.assertEqual(generated.size(1), prompt.size(1) + 2)
        self.assertEqual(generated[0, -2:].tolist(), [255, 254])


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
        prompt = f"M:{memory}\nQ:{question}\\nA:"
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

    def test_evaluator_prefers_curated_irrelevant_memories(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.evaluate_tiny_lm_memory_qa import load_cases, build_condition_prompts
        from src.ltm.tiny_lm import ByteTokenizer
        case = {
            "id": "curated",
            "question": "Which database stores memories?",
            "memories": ["Memories are stored in SQLite."],
            "irrelevant_memories": ["The model uses a decoder-only Transformer."],
            "expected_answer": "SQLite",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\n", encoding="utf-8")
            loaded = load_cases(path)
        self.assertEqual(loaded[0]["irrelevant_memories"], case["irrelevant_memories"])
        prompts = build_condition_prompts(
            case["question"], [{"content": case["memories"][0]}], 128,
            ByteTokenizer(), distractor_memories=loaded[0]["irrelevant_memories"],
        )
        self.assertIn("decoder-only Transformer", prompts["irrelevant_memory"])
        self.assertNotIn("SQLite", prompts["irrelevant_memory"])

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
        self.assertFalse(retrieval_contains_expected(
            [{"content": "The configured interval is 145 minutes."}], "45 minutes"
        ))


    def test_answer_matching_requires_complete_token_boundaries(self):
        from scripts.evaluate_tiny_lm_memory_qa import contains_normalized_answer
        self.assertTrue(contains_normalized_answer(
            "Answer: 45 minutes.", "45 minutes"
        ))
        self.assertTrue(contains_normalized_answer(
            "The format is newline-delimited JSON.", "newline-delimited JSON"
        ))
        self.assertFalse(contains_normalized_answer("145 minutes", "45 minutes"))
        self.assertFalse(contains_normalized_answer("SQLite3", "SQLite"))
        self.assertFalse(contains_normalized_answer("The answer is absent.", ""))

    def test_expanded_instruction_and_heldout_sets_are_distinct(self):
        from scripts.train_tiny_lm_sft import load_examples
        from scripts.evaluate_tiny_lm_memory_qa import load_cases
        training = load_examples("examples/tiny_lm_sft_demo.jsonl")
        heldout = load_cases("examples/tiny_lm_memory_qa_eval.jsonl")
        self.assertGreaterEqual(len(training), 40)
        self.assertGreaterEqual(len({row["prompt"] for row in training}), 40)
        self.assertGreaterEqual(len(heldout), 12)
        self.assertEqual(len({row["id"] for row in heldout}), len(heldout))

    def test_unseen_fact_evaluation_is_disjoint_from_sft_facts(self):
        from scripts.train_tiny_lm_sft import load_examples
        from scripts.evaluate_tiny_lm_memory_qa import load_cases
        training = load_examples("examples/tiny_lm_sft_demo.jsonl")
        heldout = load_cases("examples/tiny_lm_memory_qa_unseen_facts.jsonl")
        training_text = " ".join(row["prompt"] + " " + row["response"] for row in training).lower()
        self.assertEqual(len(heldout), 64)
        self.assertEqual(len({row["id"] for row in heldout}), 64)
        clusters = {row["fact_cluster_id"] for row in heldout}
        self.assertEqual(len(clusters), 32)
        self.assertTrue(all(sum(row["fact_cluster_id"] == cluster for row in heldout) == 2 for cluster in clusters))
        for row in heldout:
            self.assertNotIn(row["expected_answer"].lower(), training_text)
            self.assertNotIn(row["memories"][0].lower(), training_text)
            self.assertTrue(row["irrelevant_memories"])
            self.assertNotIn(row["expected_answer"].lower(), " ".join(row["irrelevant_memories"]).lower())

    def test_generated_memory_grounding_sft_is_deterministic_and_value_disjoint(self):
        from scripts.generate_memory_sft import build_examples
        from scripts.train_tiny_lm_sft import load_examples
        from scripts.evaluate_tiny_lm_memory_qa import load_cases
        generated = build_examples()
        self.assertEqual(len(generated), 64)
        self.assertEqual(len({row["id"] for row in generated}), 64)
        self.assertEqual(generated, build_examples())
        training = load_examples("examples/tiny_lm_sft_demo.jsonl")
        heldout = load_cases("examples/tiny_lm_memory_qa_unseen_facts.jsonl")
        training_text = " ".join(
            row["prompt"] + " " + row["response"] for row in training + generated
        ).lower()
        for case in heldout:
            self.assertNotIn(case["expected_answer"].lower(), training_text)
            self.assertNotIn(case["memories"][0].lower(), training_text)
        for row in generated:
            self.assertIn("M:", row["prompt"])
            self.assertIn("\nQ:", row["prompt"])
            self.assertTrue(row["response"].strip())

    def test_generated_answer_token_f1_reports_partial_overlap(self):
        from scripts.evaluate_tiny_lm_memory_qa import token_f1
        self.assertEqual(token_f1("SQLite database", "SQLite database"), 1.0)
        self.assertAlmostEqual(token_f1("SQLite only", "SQLite database"), 0.5)
        self.assertEqual(token_f1("unrelated output", "SQLite database"), 0.0)
        self.assertEqual(token_f1("", "SQLite"), 0.0)

    def test_generate_answer_stops_on_newline_after_prompt_boundary(self):
        import torch
        from unittest.mock import patch
        from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig, ByteTokenizer
        from scripts.evaluate_tiny_lm_memory_qa import generate_answer

        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=16, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0,
        ))
        # Force the first generated byte to newline; it should stop immediately.
        with patch("torch.argmax", return_value=torch.tensor([[10]])) as argmax:
            answer = generate_answer(
                model, ByteTokenizer(), "Q:where?\nA:", 8, "cpu"
            )
        self.assertEqual(answer, "")
        self.assertEqual(argmax.call_count, 1)


    def test_fact_cluster_bootstrap_resamples_clusters_not_paraphrases(self):
        from scripts.evaluate_tiny_lm_memory_qa import cluster_bootstrap_delta
        rows = [
            {"case_id": "a1", "fact_cluster_id": "A", "memory": {"score": 2.0}, "no_memory": {"score": 1.0}},
            {"case_id": "b1", "fact_cluster_id": "B", "memory": {"score": 0.0}, "no_memory": {"score": 1.0}},
            {"case_id": "b2", "fact_cluster_id": "B", "memory": {"score": 0.0}, "no_memory": {"score": 1.0}},
            {"case_id": "b3", "fact_cluster_id": "B", "memory": {"score": 0.0}, "no_memory": {"score": 1.0}},
        ]
        first = cluster_bootstrap_delta(rows, "memory", "no_memory", "score", replicates=500, seed=17)
        second = cluster_bootstrap_delta(rows, "memory", "no_memory", "score", replicates=500, seed=17)
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"], 2)
        self.assertAlmostEqual(first["mean_delta"], 0.0)
        self.assertEqual(first["positive_delta_clusters"], 1)
        self.assertEqual(first["negative_delta_clusters"], 1)

    def test_unseen_fact_clusters_have_two_paraphrases_each(self):
        from scripts.evaluate_tiny_lm_memory_qa import load_cases
        cases = load_cases("examples/tiny_lm_memory_qa_unseen_facts.jsonl")
        by_cluster = {}
        for case in cases:
            by_cluster.setdefault(case["fact_cluster_id"], []).append(case)
        self.assertEqual(len(by_cluster), 32)
        self.assertTrue(all(len(rows) == 2 for rows in by_cluster.values()))
        self.assertTrue(all(
            rows[0]["expected_answer"] == rows[1]["expected_answer"]
            and rows[0]["memories"] == rows[1]["memories"]
            for rows in by_cluster.values()
        ))

if __name__ == "__main__":
    unittest.main()
