"""Paired seed-matched ablation of joint-span memory-key representations."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


MODES = ("contextual", "token_embedding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--output-dir", default="joint-span-key-ablation")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("seeds must be non-empty and unique")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paired = []
    for seed in args.seeds:
        metrics = {}
        for mode in MODES:
            stem = f"seed-{seed}-{mode}"
            checkpoint = output_dir / f"{stem}.pt"
            result = output_dir / f"{stem}.json"
            log = output_dir / f"{stem}.log"
            command = [
                sys.executable, "scripts/train_tiny_lm_joint_span.py",
                "--input", args.input,
                "--checkpoint", args.checkpoint,
                "--output-checkpoint", str(checkpoint),
                "--eval-cases", args.eval_cases,
                "--output", str(result),
                "--steps", str(args.steps),
                "--batch-size", str(args.batch_size),
                "--learning-rate", str(args.learning_rate),
                "--seed", str(seed),
                "--device", args.device,
                "--memory-key-mode", mode,
            ]
            with log.open("w", encoding="utf-8") as stream:
                subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
            payload = json.loads(result.read_text(encoding="utf-8"))
            metrics[mode] = payload["after_joint_span_supervision"]
        contextual = metrics["contextual"]
        raw = metrics["token_embedding"]
        paired_row = {
            "seed": seed,
            "examples": contextual["examples"],
            "contextual_exact_span_accuracy": contextual["joint_exact_span_accuracy"],
            "token_embedding_exact_span_accuracy": raw["joint_exact_span_accuracy"],
            "exact_accuracy_delta_contextual_minus_token_embedding": (
                contextual["joint_exact_span_accuracy"] - raw["joint_exact_span_accuracy"]
            ),
            "contextual_mean_joint_span_rank": contextual["mean_joint_span_rank"],
            "token_embedding_mean_joint_span_rank": raw["mean_joint_span_rank"],
            "mean_rank_delta_contextual_minus_token_embedding": (
                contextual["mean_joint_span_rank"] - raw["mean_joint_span_rank"]
            ),
            "results": {mode: f"seed-{seed}-{mode}.json" for mode in MODES},
        }
        paired.append(paired_row)
        print(json.dumps(paired_row, sort_keys=True), flush=True)

    deltas = [row["exact_accuracy_delta_contextual_minus_token_embedding"] for row in paired]
    summary = {
        "protocol": (
            "paired per-seed runs; same copy-enabled base checkpoint, seed, training examples, "
            "optimizer settings, steps, and evaluation cases; both model and span head fine-tuned; "
            "no gold answer length; only memory-key representation differs"
        ),
        "seeds": args.seeds,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "paired_runs": paired,
        "mean_contextual_exact_span_accuracy": sum(
            row["contextual_exact_span_accuracy"] for row in paired
        ) / len(paired),
        "mean_token_embedding_exact_span_accuracy": sum(
            row["token_embedding_exact_span_accuracy"] for row in paired
        ) / len(paired),
        "mean_paired_exact_accuracy_delta": sum(deltas) / len(deltas),
        "contextual_wins": sum(delta > 0 for delta in deltas),
        "ties": sum(delta == 0 for delta in deltas),
        "token_embedding_wins": sum(delta < 0 for delta in deltas),
    }
    path = output_dir / "paired-key-ablation-summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
