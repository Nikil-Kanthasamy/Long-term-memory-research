import unittest

import torch

from scripts.train_tiny_lm_span_copy import locate_answer_span, score_answer_span
from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig


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


if __name__ == "__main__":
    unittest.main()
