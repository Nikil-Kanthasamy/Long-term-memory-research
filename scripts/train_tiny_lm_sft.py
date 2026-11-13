"""Supervised instruction-tuning for the from-scratch byte-level TinyLM.

Input is JSONL with {"prompt": "...", "response": "..."}. Loss is applied
only to response bytes (and the final newline), never to prompt bytes.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel


def load_examples(path):
    rows = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("prompt"), str) or not isinstance(item.get("response"), str):
            raise ValueError(f"line {line_no}: prompt and response must be strings")
        if not item["prompt"].strip() or not item["response"].strip():
            raise ValueError(f"line {line_no}: prompt and response cannot be empty")
        rows.append(item)
    if not rows:
        raise ValueError("dataset contains no examples")
    return rows


def encode_example(prompt, response, block_size):
    """Return input IDs and labels; labels on prompt positions are ignored."""
    tokenizer = ByteTokenizer()
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response.rstrip() + "\n")
    # Keep the answer intact when possible; trim the oldest prompt bytes first.
    if len(response_ids) >= block_size:
        response_ids = response_ids[-(block_size - 1):]
        prompt_ids = []
    else:
        prompt_ids = prompt_ids[-(block_size - len(response_ids)):]
    tokens = prompt_ids + response_ids
    if len(tokens) < 2:
        raise ValueError("encoded example must contain at least two bytes")
    inputs = tokens[:-1]
    labels = tokens[1:]
    first_response_target = max(0, len(prompt_ids) - 1)
    labels[:first_response_target] = [-100] * first_response_target
    return inputs, labels


def make_batch(rows, block_size, device):
    encoded = [encode_example(row["prompt"], row["response"], block_size) for row in rows]
    width = min(block_size - 1, max(len(x) for x, _ in encoded))
    inputs, labels = [], []
    for x, y in encoded:
        x, y = x[:width], y[:width]
        pad = width - len(x)
        inputs.append(x + [0] * pad)
        labels.append(y + [-100] * pad)
    return (torch.tensor(inputs, dtype=torch.long, device=device),
            torch.tensor(labels, dtype=torch.long, device=device))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL prompt/response examples")
    parser.add_argument("--checkpoint", required=True, help="Base randomly initialized/trained checkpoint")
    parser.add_argument("--output", default="models/tiny-memory-lm-instruct.pt")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    rows = load_examples(args.input)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for step in range(1, args.steps + 1):
        batch = random.choices(rows, k=min(args.batch_size, len(rows)))
        x, labels = make_batch(batch, model.config.block_size, device)
        logits, _ = model(x)
        loss = F.cross_entropy(
            logits.reshape(-1, model.config.vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step={step} response_only_loss={loss.item():.4f}")
    payload = model.checkpoint()
    payload.update({
        "stage": "supervised_instruction_tuning",
        "base_checkpoint_step": checkpoint.get("step"),
        "sft_steps": args.steps,
        "training_examples": len(rows),
        "loss_mask": "prompt_bytes_ignored",
    })
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"Saved instruction-tuned checkpoint to {output}")


if __name__ == "__main__":
    main()
