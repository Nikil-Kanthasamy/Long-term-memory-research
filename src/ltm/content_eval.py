"""Controlled content-generation comparisons and human-rating templates."""
import csv
import json
import random
from pathlib import Path

from .content import create_content, editorial_review

RATING_FIELDS = (
    "relevance", "personalization", "factual_consistency", "style_consistency",
    "completeness", "readability", "unsupported_claims",
)


def load_content_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Content evaluation dataset must be a non-empty JSON list")
    for index, case in enumerate(payload):
        if not isinstance(case, dict) or "topic" not in case or not str(case["topic"]).strip():
            raise ValueError(f"Case {index} must be an object with a non-empty topic")
    return payload


def run_content_evaluation(store, user_id, cases, output_dir, top_k=5,
                           include_review=True, generator=None, retriever=None,
                           random_seed=2026):
    """Generate matched drafts, a private provenance manifest, and blinded rating rows.

    Draft filenames contain only blind IDs. Keep manifest.jsonl private: it maps
    blind IDs to experimental conditions and memory provenance.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if not cases:
        raise ValueError("cases must not be empty")
    output = Path(output_dir)
    drafts_dir = output / "drafts"
    drafts_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for index, case in enumerate(cases, start=1):
        common = {key: case.get(key, default) for key, default in {
            "topic": "", "content_type": "blog", "audience": "general readers",
            "tone": "professional", "length": "medium", "keywords": [], "instructions": "",
        }.items()}
        common.update({"top_k": top_k, "generator": generator, "retriever": retriever})
        conditions = [("no_memory", False), ("memory", True)]
        if include_review:
            conditions.append(("memory_plus_review", True))
        for condition, use_memory in conditions:
            result = create_content(store, user_id, use_memory=use_memory, **common)
            blind_id = f"D{len(records) + 1:04d}"
            filename = f"{blind_id}.md"
            (drafts_dir / filename).write_text(
                result["content"].rstrip() + "\n", encoding="utf-8"
            )
            record = {
                "blind_draft_id": blind_id,
                "case_id": case.get("id", f"case-{index:03d}"),
                "condition": condition,
                "draft_file": f"drafts/{filename}",
                "content_type": result["content_type"],
                "topic": result["topic"],
                "memory_count": result["memory_count"],
                "memory_ids": result["memory_ids"],
            }
            if condition == "memory_plus_review":
                record["editorial_review"] = editorial_review(
                    result["content"], generator=generator
                )
            records.append(record)

    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # One row per draft is a starting template. Add one duplicate row per
    # additional reviewer and fill reviewer_id; do not share the manifest.
    rating_ids = [record["blind_draft_id"] for record in records]
    random.Random(random_seed).shuffle(rating_ids)
    with (output / "ratings_template.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["reviewer_id", "blind_draft_id", *RATING_FIELDS, "reviewer_notes"],
        )
        writer.writeheader()
        for blind_id in rating_ids:
            writer.writerow({"reviewer_id": "", "blind_draft_id": blind_id})

    summary = {
        "cases": len(cases),
        "drafts": len(records),
        "conditions": list(dict.fromkeys(record["condition"] for record in records)),
        "rating_fields": list(RATING_FIELDS),
        "rating_order_seed": random_seed,
        "manifest_is_private": True,
        "note": (
            "No quality scores are inferred. Complete ratings_template.csv with "
            "blinded human ratings; duplicate rows for additional reviewers."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
