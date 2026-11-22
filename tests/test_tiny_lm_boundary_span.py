import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from scripts.train_tiny_lm_boundary_span import (
        SpanBoundaryHead, boundary_loss, score_boundary_span,
    )
    from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TinyLMBoundarySpanTests(unittest.TestCase):
    def test_model_can_return_hidden_states_without_changing_default_api(self):
        torch.manual_seed(11)
        model = TinyLanguageModel(TinyLMConfig(
            vocab_size=256, block_size=32, d_model=16,
            n_heads=4, n_layers=1, dropout=0.0, copy_enabled=True,
        ))
        ids = torch.tensor([[ord(c) for c in "M:fact\nQ:what?\nA:"]])
        logits, loss, hidden = model(ids, return_hidden=True)
        self.assertEqual(tuple(logits.shape), (1, ids.shape[1], 256))
        self.assertIsNone(loss)
        self.assertEqual(tuple(hidden.shape), (1, ids.shape[1], 16))
        _, _, attention = model(ids, return_copy_attention=True)
        self.assertEqual(tuple(attention.shape), (1, ids.shape[1], ids.shape[1]))

    def test_boundary_head_emits_independent_start_and_end_scores(self):
        head = SpanBoundaryHead(16)
        start, end = head(torch.randn(2, 1, 16), torch.randn(2, 9, 16))
        self.assertEqual(tuple(start.shape), (2, 1, 9))
        self.assertEqual(tuple(end.shape), (2, 1, 9))

    def test_boundary_loss_backpropagates(self):
        start = torch.randn(1, 8, requires_grad=True)
        end = torch.randn(1, 8, requires_grad=True)
        loss = boundary_loss(start, end, torch.tensor([2]), torch.tensor([5]))
        loss.backward()
        self.assertIsNotNone(start.grad)
        self.assertIsNotNone(end.grad)

    def test_exact_span_scoring_does_not_need_gold_length_candidates(self):
        start = torch.tensor([0.1, 0.8, 0.0, 0.0, 0.0])
        end = torch.tensor([0.0, 0.0, 0.0, 0.7, 0.1])
        result = score_boundary_span(start, end, 1, 4)
        self.assertTrue(result["exact_span_match"])
        self.assertEqual(result["start_rank"], 1)
        self.assertEqual(result["end_rank"], 1)


if __name__ == "__main__":
    unittest.main()
