"""Evaluate next-byte cross-entropy and perplexity on a held-out text file."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.tiny_lm import TinyLanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Held-out UTF-8 text file")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    raw = Path(args.input).read_bytes()
    tokens = torch.tensor(list(raw), dtype=torch.long)
    block = model.config.block_size
    if len(tokens) <= block:
        parser.error("evaluation file must contain more bytes than model block_size")

    weighted_loss = 0.0
    token_count = 0
    model.eval()
    with torch.no_grad():
        for start in range(0, len(tokens) - block, block):
            x = tokens[start:start + block].unsqueeze(0).to(device)
            y = tokens[start + 1:start + block + 1].unsqueeze(0).to(device)
            logits, _ = model(x)
            count = y.numel()
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum")
            weighted_loss += float(loss)
            token_count += count
    mean_loss = weighted_loss / token_count
    try:
        perplexity = math.exp(mean_loss)
    except OverflowError:
        perplexity = None
    print(json.dumps({
        "checkpoint_step": checkpoint.get("step"),
        "evaluation_bytes": len(raw),
        "evaluated_next_bytes": token_count,
        "mean_cross_entropy_nats_per_byte": mean_loss,
        "perplexity_per_byte": perplexity,
        "device": device,
    }, indent=2))


if __name__ == "__main__":
    main()
