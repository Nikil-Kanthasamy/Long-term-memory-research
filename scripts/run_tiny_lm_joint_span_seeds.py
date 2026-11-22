"""Run a reproducible multi-seed sweep for answer-length-free joint span selection."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--output-dir", default="joint-span-seeds")
    parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in args.seeds:
        checkpoint = output_dir / f"joint-span-seed-{seed}.pt"
        result = output_dir / f"joint-span-seed-{seed}.json"
        log = output_dir / f"joint-span-seed-{seed}.log"
        command = [
            sys.executable, "scripts/train_tiny_lm_joint_span.py",
            "--input", args.input,
            "--checkpoint", args.checkpoint,
            "--output-checkpoint", str(checkpoint),
            "--eval-cases", args.eval_cases,
            "--output", str(result),
            "--steps", str(args.steps),
            "--batch-size", str(args.batch_size),
            "--seed", str(seed),
            "--device", args.device,
        ]
        with log.open("w", encoding="utf-8") as stream:
            subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT)
        payload = json.loads(result.read_text(encoding="utf-8"))
        metrics = payload["after_joint_span_supervision"]
        rows.append({
            "seed": seed,
            "examples": metrics["examples"],
            "joint_exact_span_accuracy": metrics["joint_exact_span_accuracy"],
            "start_top1_accuracy": metrics["start_top1_accuracy"],
            "end_top1_accuracy": metrics["end_top1_accuracy"],
            "mean_joint_span_rank": metrics["mean_joint_span_rank"],
            "checkpoint": str(checkpoint),
            "evaluation": str(result),
            "log": str(log),
        })
        print(json.dumps(rows[-1], sort_keys=True), flush=True)

    accuracies = [row["joint_exact_span_accuracy"] for row in rows]
    summary = {
        "protocol": "same frozen copy-enabled base checkpoint; independent joint-head initialization per seed; no gold answer length",
        "seeds": args.seeds,
        "runs": rows,
        "joint_exact_span_accuracy_mean": sum(accuracies) / len(accuracies),
        "joint_exact_span_accuracy_min": min(accuracies),
        "joint_exact_span_accuracy_max": max(accuracies),
    }
    summary_path = output_dir / "joint-span-seed-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
