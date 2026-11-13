"""Seed synthetic, non-sensitive memories for the content-evaluation pilot."""
import argparse
import json
from pathlib import Path

from src.ltm.config import db_path
from src.ltm.store import MemoryStore


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user", default="content-eval-pilot")
    parser.add_argument(
        "--memories",
        default="examples/content_eval_pilot_memories.json",
        help="JSON array of synthetic memory records",
    )
    args = parser.parse_args()
    with Path(args.memories).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError("Memory seed file must contain a non-empty JSON list")

    store = MemoryStore(db_path())
    ids = []
    for record in records:
        if not isinstance(record, dict) or not str(record.get("content", "")).strip():
            raise ValueError("Every seed record must contain non-empty content")
        ids.append(store.add_memory(
            args.user,
            record["content"],
            category=record.get("category", "fact"),
            importance=record.get("importance", 0.5),
            confidence=record.get("confidence", 1.0),
        ))
    print(json.dumps({
        "user_id": args.user,
        "seed_records": len(records),
        "memory_ids": ids,
        "unique_memory_ids": len(set(ids)),
        "database": db_path(),
        "note": "Synthetic data only. Re-running is idempotent for exact duplicate memory text.",
    }, indent=2))


if __name__ == "__main__":
    main()
