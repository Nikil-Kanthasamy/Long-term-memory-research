"""Sweep fixed span-width penalties over a held-out development set; performs no training."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_tiny_lm_joint_span import JointSpanHead, evaluate
from scripts.train_tiny_lm_span_copy import load_jsonl
from src.ltm.tiny_lm import TinyLanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Joint-span checkpoint produced by train_tiny_lm_joint_span.py")
    parser.add_argument("--dev-cases", required=True, help="Fact-cluster-held-out development JSONL; never use the final test set here")
    parser.add_argument("--penalties", type=float, nargs="+", default=[0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0])
    parser.add_argument("--output", default="joint-span-width-penalty-sweep.json")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if not args.penalties or any(not math.isfinite(value) or value < 0 for value in args.penalties):
        parser.error("penalties must be a non-empty list of finite, non-negative values")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint["model"], map_location=args.device)
    head = JointSpanHead(model.config.d_model).to(args.device)
    head.load_state_dict(checkpoint["joint_span_head"])
    cases = load_jsonl(args.dev_cases)

    results = []
    for penalty in args.penalties:
        metrics = evaluate(
            model, head, cases, args.device, model.config.block_size,
            span_width_penalty=penalty,
        )
        results.append({
            "span_width_penalty_per_byte": penalty,
            "examples": metrics["examples"],
            "skipped_missing_answer_span": metrics["skipped_missing_answer_span"],
            "joint_exact_span_accuracy": metrics["joint_exact_span_accuracy"],
            "mean_joint_span_rank": metrics["mean_joint_span_rank"],
            "per_case": metrics["per_case"],
        })
    report = {
        "checkpoint": args.checkpoint,
        "development_cases": args.dev_cases,
        "protocol": "post-hoc decoding sweep only; no model updates; all ordered spans start<=end; penalty is fixed per predicted byte; choose penalty on development facts only",
        "results": results,
        "selection_rule": "select the penalty with highest development exact-span accuracy; break ties by lower mean joint-span rank, then lower penalty",
    }
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
