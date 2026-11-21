import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from scripts.train_tiny_lm_span_copy import evaluate, fit_memory_prompt, locate_answer_span, prepare_training_examples, score_answer_span
    from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TinyLMSpanCopyTests(unittest.TestCase):
    def test_forward_can_return_causal_copy_attention(self):
        torch.manual_seed(3)
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0, copy_enabled=True,
        ))
        ids = torch.tensor([[ord(c) for c in "M:CBOR\nQ:format?\nA:"]])
        _, _, attention = model(ids, return_copy_attention=True)
        self.assertEqual(tuple(attention.shape), (1, ids.shape[1], ids.shape[1]))
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones(1, ids.shape[1]), atol=1e-6))
        self.assertEqual(float(attention[0, 0, 1:].sum()), 0.0)

    def test_answer_span_offsets_and_memory_boundary(self):
        ids, start, end, query, memory_end = locate_answer_span(
            "M:The export format is CBOR.\nQ:Which format?\nA:", "CBOR"
        )
        self.assertEqual(bytes(ids[start:end]).decode(), "CBOR")
        self.assertLess(end, memory_end)
        self.assertEqual(query, len(ids) - 1)

    def test_evaluation_prompt_truncation_preserves_answer_span(self):
        memory = "A long prefix with details. " * 4 + "The export format is CBOR. " + "Extra context. " * 4
        prompt = fit_memory_prompt("Which format?", memory, "CBOR", block_size=64)
        ids, start, end, _, memory_end = locate_answer_span(prompt, "CBOR")
        self.assertLessEqual(len(ids), 64)
        self.assertEqual(bytes(ids[start:end]).decode(), "CBOR")
        self.assertLessEqual(end, memory_end)

    def test_evaluation_prompt_rejects_answer_missing_from_memory(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            fit_memory_prompt("Which format?", "Format is JSON.", "CBOR", block_size=64)

    def test_span_rank_uses_equal_length_candidate_windows(self):
        attention = torch.tensor([0.0, 0.0, 0.1, 0.7, 0.1, 0.1, 0.0])
        result = score_answer_span(attention, 3, 4, 2, 6)
        self.assertAlmostEqual(result["gold_span_mass"], 0.7)
        self.assertEqual(result["gold_span_rank"], 1)
        self.assertTrue(result["top_span_match"])
        self.assertEqual(result["candidate_spans"], 4)

    def test_missing_answer_span_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not present"):
            locate_answer_span("M:export=CBOR\nQ:format?\nA:", "JSON")

    def test_evaluation_skips_non_extractive_paraphrases_and_reports_count(self):
        torch.manual_seed(5)
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0, copy_enabled=True,
        ))
        cases = [
            {"id": "extractive", "question": "Which format?", "expected_answer": "CBOR",
             "memories": ["The export format is CBOR."]},
            {"id": "paraphrase", "question": "Which format?", "expected_answer": "It uses CBOR.",
             "memories": ["The export format is CBOR."]},
        ]
        result = evaluate(model, cases, "cpu", 64)
        self.assertEqual(result["examples"], 1)
        self.assertEqual(result["skipped_missing_answer_span"], 1)
        self.assertEqual(result["per_case"][0]["case_id"], "extractive")

    def test_evaluation_rejects_dataset_with_no_extractable_answers(self):
        torch.manual_seed(6)
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=64, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0, copy_enabled=True,
        ))
        cases = [{"question": "Which format?", "expected_answer": "JSON",
                  "memories": ["The export format is CBOR."]}]
        with self.assertRaisesRegex(ValueError, "no evaluation cases contain"):
            evaluate(model, cases, "cpu", 64)

    def test_training_preparation_skips_paraphrases_without_spans(self):
        rows = [
            {"prompt": "M:The export format is CBOR.\nQ:format?\nA:", "response": "CBOR"},
            {"prompt": "M:The export format is CBOR.\nQ:format?\nA:", "response": "It uses CBOR."},
        ]
        prepared, skipped_missing, skipped_long = prepare_training_examples(
            rows, block_size=64
        )
        self.assertEqual(len(prepared), 1)
        self.assertEqual(skipped_missing, 1)
        self.assertEqual(skipped_long, 0)



if __name__ == "__main__":
    unittest.main()
