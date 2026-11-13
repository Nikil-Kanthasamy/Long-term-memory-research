"""Train a small byte-level GPT-style model from scratch on a plain-text corpus."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import torch

from src.ltm.tiny_lm import TinyLanguageModel, TinyLMConfig


def batches(data, batch_size, block_size, device):
    starts = torch.randint(0, len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in starts]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts]).to(device)
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="UTF-8 plain-text training corpus")
    parser.add_argument("--validation-input", help="Separate held-out validation text file")
    parser.add_argument("--output-dir", default="models/tiny-memory-lm")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--eval-interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1 or args.block_size < 8:
        parser.error("steps and batch-size must be positive; block-size must be >= 8")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device

    corpus_path = Path(args.input)
    raw = corpus_path.read_bytes()
    if len(raw) < args.block_size * 4:
        parser.error("corpus is too small; provide at least 4 * block-size bytes")
    train_data = torch.tensor(list(raw), dtype=torch.long)
    if args.validation_input:
        validation_raw = Path(args.validation_input).read_bytes()
        val_data = torch.tensor(list(validation_raw), dtype=torch.long)
    else:
        split = int(len(train_data) * 0.9)
        val_data = train_data[split:]
        train_data = train_data[:split]
        validation_raw = bytes(val_data.tolist())
    if len(val_data) <= args.block_size or len(train_data) <= args.block_size:
        parser.error("training and validation files must each contain more than block-size bytes")
    validation_sha256 = hashlib.sha256(validation_raw).hexdigest()

    config = TinyLMConfig(
        block_size=args.block_size, d_model=args.d_model,
        n_heads=args.heads, n_layers=args.layers,
    )
    model = TinyLanguageModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    def estimate(data_source, count=20):
        model.eval()
        values = []
        with torch.no_grad():
            for _ in range(count):
                x, y = batches(data_source, args.batch_size, args.block_size, device)
                _, loss = model(x, y)
                values.append(float(loss))
        model.train()
        return sum(values) / len(values)

    print(json.dumps({
        "device": device, "training_bytes": len(raw), "validation_bytes": len(validation_raw),
        "parameters": sum(p.numel() for p in model.parameters()),
        "config": config.__dict__,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "validation_sha256": validation_sha256,
    }, indent=2))
    for step in range(1, args.steps + 1):
        model.train()
        x, y = batches(train_data, args.batch_size, args.block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step == 1 or step % args.eval_interval == 0 or step == args.steps:
            train_loss = estimate(train_data, count=5)
            val_loss = estimate(val_data, count=10)
            print(f"step={step} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
            if val_loss < best_val:
                best_val = val_loss
                torch.save({
                    **model.checkpoint(),
                    "step": step,
                    "best_val_loss": best_val,
                    "corpus_sha256": hashlib.sha256(raw).hexdigest(),
                    "validation_sha256": validation_sha256,
                    "tokenizer": "utf8-byte-v1",
                }, output_dir / "best.pt")
    torch.save({
        **model.checkpoint(),
        "step": args.steps,
        "best_val_loss": best_val,
        "corpus_sha256": hashlib.sha256(raw).hexdigest(),
        "validation_sha256": validation_sha256,
        "tokenizer": "utf8-byte-v1",
    }, output_dir / "final.pt")
    print(f"Saved checkpoints to {output_dir}")


if __name__ == "__main__":
    main()
