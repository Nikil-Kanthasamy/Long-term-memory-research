"""Small decoder-only Transformer language model trained from random initialization.

This module is optional: install requirements-tiny-lm.txt to use it. The
byte tokenizer deliberately has a fixed 256-token vocabulary, avoiding any
pretrained tokenizer or downloaded model weights.
"""
from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class TinyLMConfig:
    vocab_size: int = 256
    block_size: int = 256
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    dropout: float = 0.1


class ByteTokenizer:
    """UTF-8 byte tokenizer with no learned or downloaded vocabulary."""

    vocab_size = 256

    @staticmethod
    def encode(text):
        return list(text.encode("utf-8"))

    @staticmethod
    def decode(token_ids):
        return bytes(max(0, min(255, int(token))) for token in token_ids).decode(
            "utf-8", errors="replace"
        )


class TinyLanguageModel(nn.Module):
    """Compact GPT-style causal Transformer; all weights start randomly."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config or TinyLMConfig()
        c = self.config
        if c.d_model % c.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.token_embedding = nn.Embedding(c.vocab_size, c.d_model)
        self.position_embedding = nn.Embedding(c.block_size, c.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=c.d_model,
            nhead=c.n_heads,
            dim_feedforward=4 * c.d_model,
            dropout=c.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(
            layer, num_layers=c.n_layers, enable_nested_tensor=False
        )
        self.final_norm = nn.LayerNorm(c.d_model)
        self.output = nn.Linear(c.d_model, c.vocab_size, bias=False)
        self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, targets=None):
        batch, length = input_ids.shape
        if length > self.config.block_size:
            raise ValueError("input exceeds configured block_size")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)
        # True entries are masked by PyTorch: tokens cannot attend to the future.
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=input_ids.device),
            diagonal=1,
        )
        hidden = self.blocks(hidden, mask=causal_mask)
        logits = self.output(self.final_norm(hidden))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=40):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.eval()
        for _ in range(max_new_tokens):
            context = input_ids[:, -self.config.block_size:]
            logits, _ = self(context)
            logits = logits[:, -1, :] / temperature
            if top_k and top_k > 0:
                count = min(int(top_k), logits.size(-1))
                threshold = torch.topk(logits, count).values[:, [-1]]
                logits = logits.masked_fill(logits < threshold, float("-inf"))
            probabilities = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            input_ids = torch.cat((input_ids, next_token), dim=1)
        return input_ids

    def checkpoint(self):
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location="cpu"):
        config = TinyLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location)
