"""Offline error analysis for answer-length-free joint-span evaluation JSONs."""
import argparse
import json
from pathlib import Path
from statistics import mean


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected an object")
        rows.append(row)
    return rows


def gold_span(case, block_size):
    """Return gold byte offsets in the same answer-independent retained memory."""
    memory = (case.get("memories") or [None])[0]
    if memory is None:
        raise ValueError(f"{case.get('id')}: missing memory")
    memory_bytes = memory.encode("utf-8")
    question_suffix = f"\nQ:{case['question']}\nA:".encode("utf-8")
    memory_budget = block_size - 2 - len(question_suffix)
    if memory_budget < 0:
        raise ValueError(f"{case.get('id')}: question exceeds context")
    retained = memory_bytes[:memory_budget]
    answer = case["expected_answer"].encode("utf-8")
    start = retained.find(answer)
    if start < 0:
        return None
    return start, start + len(answer) - 1, len(answer)


def summarize(rows):
    if not rows:
        return {"examples": 0}
    return {
        "examples": len(rows),
        "exact_span_accuracy": mean(row["exact"] for row in rows),
        "joint_start_boundary_accuracy": mean(row["start_error"] == 0 for row in rows),
        "joint_end_boundary_accuracy": mean(row["end_error"] == 0 for row in rows),
        "mean_absolute_start_error_bytes": mean(row["abs_start_error"] for row in rows),
        "mean_absolute_end_error_bytes": mean(row["abs_end_error"] for row in rows),
        "mean_signed_start_error_bytes": mean(row["start_error"] for row in rows),
        "fraction_start_too_early": mean(row["start_error"] > 0 for row in rows),
        "fraction_start_too_late": mean(row["start_error"] < 0 for row in rows),
        "mean_predicted_span_length_error_bytes": mean(row["length_error"] for row in rows),
        "fraction_predicted_span_too_long": mean(row["length_error"] > 0 for row in rows),
        "fraction_predicted_span_too_short": mean(row["length_error"] < 0 for row in rows),
    }


def analyze(cases, result_paths, block_size):
    case_by_id = {case["id"]: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    output = {"block_size": block_size, "seeds": {}, "pooled_across_seeds": {}}
    pooled = []
    for path in result_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        metrics = payload["after_joint_span_supervision"]
        seed_rows = []
        for prediction in metrics["per_case"]:
            case_id = prediction["case_id"]
            if case_id not in case_by_id:
                raise ValueError(f"{path}: unknown case ID {case_id}")
            span = gold_span(case_by_id[case_id], block_size)
            if span is None:
                raise ValueError(f"{path}: {case_id} has no answer in retained memory")
            gold_start, gold_end, gold_width = span
            predicted_start = prediction["predicted_start"]
            predicted_end = prediction["predicted_end_inclusive"]
            start_error = predicted_start - gold_start
            end_error = predicted_end - gold_end
            row = {
                "case_id": case_id,
                "gold_width_bytes": gold_width,
                "predicted_width_bytes": predicted_end - predicted_start + 1,
                "start_error": start_error,
                "end_error": end_error,
                "abs_start_error": abs(start_error),
                "abs_end_error": abs(end_error),
                "length_error": (predicted_end - predicted_start + 1) - gold_width,
                "exact": predicted_start == gold_start and predicted_end == gold_end,
            }
            seed_rows.append(row)
        seed = Path(path).stem
        output["seeds"][seed] = summarize(seed_rows)
        pooled.extend(seed_rows)

    buckets = {"1-4": [], "5-10": [], "11+": []}
    for row in pooled:
        width = row["gold_width_bytes"]
        key = "1-4" if width <= 4 else "5-10" if width <= 10 else "11+"
        buckets[key].append(row)
    output["pooled_across_seeds"] = summarize(pooled)
    output["per_case_pooled"] = pooled
    output["pooled_by_gold_answer_bytes"] = {
        key: summarize(rows) for key, rows in buckets.items()
    }
    output["warning"] = (
        "Pooled rows reuse the same evaluation cases across seeds and may include "
        "paraphrase variants from the same fact cluster; they are not independent samples."
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", required=True, help="Unseen-facts evaluation JSONL")
    parser.add_argument("--seed-results", nargs="+", required=True, help="Per-seed evaluation JSON files")
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--output", default="joint-span-error-analysis.json")
    args = parser.parse_args()
    report = analyze(load_jsonl(args.cases), args.seed_results, args.block_size)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
