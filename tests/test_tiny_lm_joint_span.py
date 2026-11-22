import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from scripts.train_tiny_lm_joint_span import (
        JointSpanHead, joint_span_loss, mask_invalid_spans, score_joint_span,
    )


@unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
class TinyLMJointSpanTests(unittest.TestCase):
    def test_joint_head_scores_all_start_end_pairs(self):
        head = JointSpanHead(16)
        start, end, pair = head(torch.randn(2, 1, 16), torch.randn(2, 9, 16))
        self.assertEqual(tuple(start.shape), (2, 9))
        self.assertEqual(tuple(end.shape), (2, 9))
        self.assertEqual(tuple(pair.shape), (2, 9, 9))

    def test_mask_excludes_reversed_spans(self):
        logits = torch.zeros(1, 4, 4)
        masked = mask_invalid_spans(logits)
        self.assertTrue(torch.isfinite(masked[0, 0, 0]))
        self.assertTrue(masked[0, 3, 1] < -1e20)
        self.assertTrue(masked[0, 1, 3] == 0)

    def test_joint_loss_backpropagates(self):
        logits = torch.randn(2, 5, 5, requires_grad=True)
        loss = joint_span_loss(mask_invalid_spans(logits), torch.tensor([1, 2]), torch.tensor([3, 4]))
        loss.backward()
        self.assertIsNotNone(logits.grad)

    def test_joint_scoring_is_answer_length_free(self):
        logits = torch.full((1, 5, 5), -10.0)
        logits[0, 1, 3] = 10.0
        result = score_joint_span(logits, 1, 3)
        self.assertTrue(result["exact_span_match"])
        self.assertEqual(result["joint_rank"], 1)


if __name__ == "__main__":
    unittest.main()
