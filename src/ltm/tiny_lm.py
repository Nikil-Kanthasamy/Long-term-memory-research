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
    copy_enabled: bool = False


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


def build_memory_prompt(question, memories, block_size, include_memory=True, reserve_new_tokens=32, tokenizer=None):
    """Fit memory and question into the byte context without dropping the memory prefix.

    The question is prioritized; remaining context is allocated to the leading
    bytes of retrieved memory. This prevents a long prompt header from silently
    evicting all evidence when the caller truncates to the model context window.
    """
    tokenizer = tokenizer or ByteTokenizer()
    question_ids = tokenizer.encode(f"Q:{question}\nA:")
    if len(question_ids) >= block_size:
        question_ids = question_ids[-block_size:]
    # Leave room for generated answer bytes so retrieved evidence does not
    # immediately slide out of the causal context window during decoding.
    generation_reserve = min(max(0, reserve_new_tokens), block_size // 2)
    prompt_budget = max(1, block_size - generation_reserve)
    if len(question_ids) > prompt_budget:
        question_ids = question_ids[-prompt_budget:]
    available = prompt_budget - len(question_ids)
    memory_ids = []
    if include_memory and memories and available > 0:
        memory_text = "M:" + "\n".join(str(item) for item in memories) + "\n"
        memory_ids = tokenizer.encode(memory_text)[:available]
    return tokenizer.decode(memory_ids + question_ids)


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
        if c.copy_enabled:
            # Pointer head: query each prediction position against token embeddings
            # already present in its causal context, then scatter attention mass
            # onto the corresponding vocabulary IDs.
            self.copy_query = nn.Linear(c.d_model, c.d_model, bias=False)
            self.copy_gate = nn.Linear(c.d_model, 1)
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
        normalized_hidden = self.final_norm(hidden)
        logits = self.output(normalized_hidden)
        if self.config.copy_enabled:
            keys = self.token_embedding(input_ids)
            queries = self.copy_query(normalized_hidden)
            copy_scores = torch.matmul(queries, keys.transpose(1, 2))
            copy_scores = copy_scores / (self.config.d_model ** 0.5)
            copy_scores = copy_scores.masked_fill(causal_mask.unsqueeze(0), float("-inf"))
            copy_attention = F.softmax(copy_scores, dim=-1)
            copy_probs = torch.zeros_like(logits).scatter_add(
                dim=-1,
                index=input_ids.unsqueeze(1).expand(-1, length, -1),
                src=copy_attention,
            )
            generation_gate = torch.sigmoid(self.copy_gate(normalized_hidden))
            vocab_probs = F.softmax(logits, dim=-1)
            mixed_probs = generation_gate * vocab_probs + (1.0 - generation_gate) * copy_probs
            logits = torch.log(mixed_probs.clamp_min(1e-9))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch * length, self.config.vocab_size),
                targets.reshape(batch * length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=200, temperature=0.8, top_k=40, stop_sequence_ids=None):
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
            if stop_sequence_ids:
                stop = [int(token) for token in stop_sequence_ids]
                if stop and input_ids.size(1) >= len(stop):
                    suffixes = input_ids[:, -len(stop):].tolist()
                    if all(suffix == stop for suffix in suffixes):
                        break
        return input_ids

    def checkpoint(self):
        return {"config": asdict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, checkpoint, map_location="cpu"):
        config = TinyLMConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        return model.to(map_location)


def tokenizer_from_checkpoint(checkpoint):
    """Restore the tokenizer recorded with a model checkpoint."""
    metadata = checkpoint.get("tokenizer", "utf8-byte-v1")
    if isinstance(metadata, dict) and metadata.get("type") == "byte-bpe-v1":
        from src.ltm.bpe_tokenizer import ByteBPETokenizer
        return ByteBPETokenizer.from_dict(metadata)
    return ByteTokenizer()
