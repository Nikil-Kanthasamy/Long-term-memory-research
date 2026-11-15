"""Deterministic byte-level BPE tokenizer trained from local text; no pretrained assets."""
from collections import Counter
import re


class ByteBPETokenizer:
    """UTF-8 byte fallback plus learned adjacent-byte merges."""

    _pattern = re.compile(r"\w+|[^\w\s]+|\s+", re.UNICODE)

    def __init__(self, merges=None):
        self.merges = [tuple(map(int, pair)) for pair in (merges or [])]
        self.ranks = {pair: 256 + rank for rank, pair in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    @classmethod
    def train(cls, text, vocab_size=512, min_pair_frequency=2):
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")
        counts = Counter(cls._pattern.findall(text))
        sequences = {
            tuple(token.encode("utf-8")): frequency
            for token, frequency in counts.items()
        }
        merges = []
        while 256 + len(merges) < vocab_size:
            pair_counts = Counter()
            for sequence, frequency in sequences.items():
                for pair in zip(sequence, sequence[1:]):
                    pair_counts[pair] += frequency
            if not pair_counts:
                break
            pair, frequency = min(
                pair_counts.items(), key=lambda item: (-item[1], item[0])
            )
            if frequency < min_pair_frequency:
                break
            new_id = 256 + len(merges)
            merges.append(pair)
            updated = {}
            for sequence, count in sequences.items():
                result = []
                index = 0
                while index < len(sequence):
                    if index + 1 < len(sequence) and sequence[index:index + 2] == pair:
                        result.append(new_id)
                        index += 2
                    else:
                        result.append(sequence[index])
                        index += 1
                updated[tuple(result)] = updated.get(tuple(result), 0) + count
            sequences = updated
        return cls(merges)

    def encode(self, text):
        output = []
        for piece in self._pattern.findall(text):
            sequence = list(piece.encode("utf-8"))
            while len(sequence) > 1:
                best_rank = None
                best_index = None
                for index, pair in enumerate(zip(sequence, sequence[1:])):
                    rank = self.ranks.get(pair)
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank, best_index = rank, index
                if best_index is None:
                    break
                sequence[best_index:best_index + 2] = [best_rank]
            output.extend(sequence)
        return output

    def decode(self, token_ids):
        pieces = [bytes([int(token)]) if 0 <= int(token) < 256 else b""
                  for token in token_ids]
        expansions = {256 + i: pair for i, pair in enumerate(self.merges)}
        cache = {}

        def expand(token):
            if token < 256:
                return bytes([token])
            if token not in cache:
                left, right = expansions[token]
                cache[token] = expand(left) + expand(right)
            return cache[token]

        return b"".join(expand(int(token)) for token in token_ids).decode(
            "utf-8", errors="replace"
        )

    def to_dict(self):
        return {"type": "byte-bpe-v1", "vocab_size": self.vocab_size,
                "merges": [list(pair) for pair in self.merges]}

    @classmethod
    def from_dict(cls, payload):
        if payload.get("type") != "byte-bpe-v1":
            raise ValueError("unsupported tokenizer format")
        tokenizer = cls(payload.get("merges", []))
        if tokenizer.vocab_size != payload.get("vocab_size"):
            raise ValueError("tokenizer vocabulary size does not match merges")
        return tokenizer
