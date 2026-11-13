"""Summarize blinded content ratings and paired condition differences."""
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

SCALES = (
    "relevance", "personalization", "factual_consistency",
    "style_consistency", "completeness", "readability",
)
COUNT_FIELDS = ("unsupported_claims",)
CONDITIONS = ("no_memory", "memory", "memory_plus_review")
PAIRS = (("memory", "no_memory"), ("memory_plus_review", "memory"))


def _number(value, field, row_number):
    if value is None or not str(value).strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Row {row_number}: {field} must be numeric") from exc
    if not math.isfinite(number):
        raise ValueError(f"Row {row_number}: {field} must be finite")
    if field in SCALES and (not 1 <= number <= 5 or not number.is_integer()):
        raise ValueError(f"Row {row_number}: {field} must be an integer from 1 to 5")
    if field in COUNT_FIELDS and (number < 0 or not number.is_integer()):
        raise ValueError(f"Row {row_number}: {field} must be a non-negative integer")
    return number


def load_manifest(path):
    records = {}
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            blind_id = record.get("blind_draft_id")
            if not blind_id or blind_id in records:
                raise ValueError(f"Manifest line {line_number}: missing or duplicate blind_draft_id")
            if record.get("condition") not in CONDITIONS:
                raise ValueError(f"Manifest line {line_number}: unknown condition")
            records[blind_id] = record
    if not records:
        raise ValueError("Manifest is empty")
    return records


def analyze_ratings(ratings_path, manifest_path):
    manifest = load_manifest(manifest_path)
    rows = []
    seen_reviewer_draft = set()
    with Path(ratings_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"blind_draft_id", *SCALES, *COUNT_FIELDS}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Ratings CSV is missing columns: " + ", ".join(sorted(missing)))
        for row_number, row in enumerate(reader, start=2):
            blind_id = (row.get("blind_draft_id") or "").strip()
            if not blind_id:
                raise ValueError(f"Row {row_number}: blind_draft_id is required")
            if blind_id not in manifest:
                raise ValueError(f"Row {row_number}: unknown blind_draft_id {blind_id}")
            values = {
                field: _number(row.get(field), field, row_number)
                for field in (*SCALES, *COUNT_FIELDS)
            }
            if not any(value is not None for value in values.values()):
                continue
            reviewer = (row.get("reviewer_id") or "").strip()
            if reviewer:
                key = (reviewer, blind_id)
                if key in seen_reviewer_draft:
                    raise ValueError(f"Row {row_number}: duplicate rating by reviewer {reviewer} for {blind_id}")
                seen_reviewer_draft.add(key)
            rows.append({
                "blind_draft_id": blind_id,
                "reviewer_id": reviewer or None,
                "case_id": str(manifest[blind_id].get("case_id", blind_id)),
                "condition": manifest[blind_id]["condition"],
                **values,
            })

    if not rows:
        raise ValueError("No completed ratings found")

    per_condition = {}
    per_draft = defaultdict(lambda: defaultdict(dict))
    reviewer_ids = set()
    for row in rows:
        condition = row["condition"]
        per_draft[row["case_id"]][condition].setdefault(
            row["blind_draft_id"], []
        ).append(row)
        if row["reviewer_id"]:
            reviewer_ids.add(row["reviewer_id"])

    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row["condition"] == condition]
        if not condition_rows:
            continue
        stats = {}
        for field in (*SCALES, *COUNT_FIELDS):
            values = [row[field] for row in condition_rows if row[field] is not None]
            stats[field] = {
                "n_ratings": len(values),
                "missing": len(condition_rows) - len(values),
                "mean": statistics.mean(values) if values else None,
                "stdev": statistics.stdev(values) if len(values) > 1 else (0.0 if values else None),
            }
        per_condition[condition] = {
            "rated_drafts": len({row["blind_draft_id"] for row in condition_rows}),
            "rating_rows": len(condition_rows),
            "metrics": stats,
        }

    # Average multiple reviewers per draft, then compare matched cases.
    draft_means = defaultdict(dict)
    for case_id, condition_map in per_draft.items():
        for condition, drafts in condition_map.items():
            for blind_id, draft_rows in drafts.items():
                for field in (*SCALES, *COUNT_FIELDS):
                    values = [row[field] for row in draft_rows if row[field] is not None]
                    if values:
                        draft_means[case_id].setdefault(condition, {}).setdefault(field, []).append(
                            statistics.mean(values)
                        )

    paired = {}
    for left, right in PAIRS:
        pair_name = f"{left}_minus_{right}"
        metric_results = {}
        for field in (*SCALES, *COUNT_FIELDS):
            differences = []
            for case_id, condition_map in draft_means.items():
                left_values = condition_map.get(left, {}).get(field, [])
                right_values = condition_map.get(right, {}).get(field, [])
                if left_values and right_values:
                    differences.append(
                        statistics.mean(left_values) - statistics.mean(right_values)
                    )
            metric_results[field] = {
                "n_paired_cases": len(differences),
                "mean_difference": statistics.mean(differences) if differences else None,
                "positive_cases": sum(value > 0 for value in differences),
                "negative_cases": sum(value < 0 for value in differences),
                "ties": sum(value == 0 for value in differences),
                "interpretation": (
                    "lower is preferable" if field in COUNT_FIELDS
                    else "higher is preferable"
                ),
            }
        paired[pair_name] = metric_results

    # Descriptive agreement only; not chance-corrected inter-rater reliability.
    agreement = {}
    for field in (*SCALES, *COUNT_FIELDS):
        by_draft = defaultdict(list)
        for row in rows:
            if row[field] is not None and row["reviewer_id"]:
                by_draft[row["blind_draft_id"]].append(row[field])
        exact_pairs = within_one_pairs = total_pairs = 0
        for values in by_draft.values():
            for i, first in enumerate(values):
                for second in values[i + 1:]:
                    total_pairs += 1
                    exact_pairs += first == second
                    within_one_pairs += abs(first - second) <= 1
        agreement[field] = {
            "reviewer_pairs": total_pairs,
            "exact_agreement": exact_pairs / total_pairs if total_pairs else None,
            "within_one_agreement": within_one_pairs / total_pairs if total_pairs else None,
            "note": "Descriptive pairwise agreement only; not a chance-corrected reliability statistic.",
        }

    return {
        "rated_rows": len(rows),
        "rated_drafts": len({row["blind_draft_id"] for row in rows}),
        "reviewers_identified": len(reviewer_ids),
        "per_condition": per_condition,
        "paired_comparisons": paired,
        "inter_rater_agreement": agreement,
        "limitations": [
            "Only completed rating rows are summarized; missing values are excluded per metric.",
            "Paired comparisons average reviewers within each draft and compare drafts matched by case_id.",
            "Descriptive agreement is not a substitute for a formal chance-corrected reliability statistic.",
            "These summaries do not establish causality or statistical significance.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ratings", required=True, help="Completed ratings CSV")
    parser.add_argument("--manifest", required=True, help="Private manifest.jsonl")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()
    report = analyze_ratings(args.ratings, args.manifest)
    rendered = json.dumps(report, indent=2)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
