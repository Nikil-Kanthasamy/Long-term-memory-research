"""Controlled content-generation comparisons and human-rating templates."""
import csv
import json
from pathlib import Path
from .content import create_content, editorial_review

RATING_FIELDS = ("relevance", "personalization", "factual_consistency", "style_consistency", "completeness", "readability", "unsupported_claims")


def load_content_cases(path):
    with Path(path).open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not payload:
        raise ValueError("Content evaluation dataset must be a non-empty JSON list")
    for index, case in enumerate(payload):
        if not isinstance(case, dict) or "topic" not in case or not str(case["topic"]).strip():
            raise ValueError(f"Case {index} must be an object with a non-empty topic")
    return payload


def run_content_evaluation(store, user_id, cases, output_dir, top_k=5, include_review=True, generator=None, retriever=None):
    """Generate matched drafts, provenance manifest, and a human-rating template."""
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
            filename = f"case-{index:03d}-{condition}.md"
            (drafts_dir / filename).write_text(result["content"].rstrip() + "\n", encoding="utf-8")
            record = {"case_id": case.get("id", f"case-{index:03d}"), "condition": condition,
                "draft_file": f"drafts/{filename}", "content_type": result["content_type"],
                "topic": result["topic"], "memory_count": result["memory_count"], "memory_ids": result["memory_ids"]}
            if condition == "memory_plus_review":
                record["editorial_review"] = editorial_review(result["content"], generator=generator)
            records.append(record)
    with (output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    # Sequential blind IDs are a template; coordinators should randomize reviewer order and keep manifest private.
    with (output / "ratings_template.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["blind_draft_id", *RATING_FIELDS, "reviewer_notes"])
        writer.writeheader()
        for number in range(1, len(records) + 1):
            writer.writerow({"blind_draft_id": f"D{number:04d}"})
    summary = {"cases": len(cases), "drafts": len(records),
        "conditions": list(dict.fromkeys(record["condition"] for record in records)),
        "rating_fields": list(RATING_FIELDS),
        "note": "No quality scores are inferred; complete ratings_template.csv with blinded human ratings."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
