import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("torch"), "optional PyTorch dependency is not installed")
class TinyLanguageModelTests(unittest.TestCase):
    def test_byte_tokenizer_round_trip(self):
        from src.ltm.tiny_lm import ByteTokenizer
        sample = "Memory works: café 🧠"
        self.assertEqual(ByteTokenizer.decode(ByteTokenizer.encode(sample)), sample)

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


if __name__ == "__main__":
    unittest.main()
