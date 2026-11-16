"""Build a deterministic, value-disjoint memory-grounding SFT corpus.

The generated records teach the model to copy a value from an explicitly
provided memory, using varied question wording. Values are synthetic and are
kept separate from the unseen-fact evaluation set.
"""
import argparse
import json
from pathlib import Path

FACTS = [
    ("cache lifetime", "17 minutes", "The cache lifetime is 17 minutes."),
    ("export format", "CBOR", "Exports are serialized in CBOR format."),
    ("replica region", "ap-south-2", "The replica is hosted in region ap-south-2."),
    ("retry limit", "7 attempts", "The retry limit is 7 attempts."),
    ("workspace code", "ORBIT-731", "The workspace code is ORBIT-731."),
    ("snapshot cadence", "every 23 hours", "Snapshots are created every 23 hours."),
    ("queue name", "copper-fern", "The processing queue is named copper-fern."),
    ("shard count", "13 shards", "The index is divided into 13 shards."),
    ("checksum method", "BLAKE2b", "Files are verified with BLAKE2b."),
    ("archive tier", "cold-glacier", "The archive tier is cold-glacier."),
    ("lease duration", "41 seconds", "A lease remains valid for 41 seconds."),
    ("schema revision", "rev-208", "The active schema revision is rev-208."),
    ("compression mode", "zstandard", "Stored payloads use zstandard compression."),
    ("worker pool", "willow-pool", "Background jobs run in willow-pool."),
    ("page size", "37 records", "Each page contains 37 records."),
    ("rotation window", "11 days", "Signing keys rotate every 11 days."),
]

QUESTION_TEMPLATES = (
    "What is the {field}?",
    "Which value is configured for the {field}?",
    "Tell me the {field} recorded in memory.",
    "According to the saved note, specify the {field}.",
)


def build_examples():
    rows = []
    for index, (field, value, memory) in enumerate(FACTS):
        for offset, template in enumerate(QUESTION_TEMPLATES):
            question = template.format(field=field)
            rows.append({
                "prompt": f"M:{memory}\\nQ:{question}\\nA:",
                "response": value,
                "id": f"grounding-{index:02d}-{offset:02d}",
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="examples/tiny_lm_sft_demo.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base)
    base = [json.loads(line) for line in base_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = base + build_examples()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\\n" for row in rows), encoding="utf-8")
    print(json.dumps({"base_examples": len(base), "generated_grounding_examples": len(rows) - len(base), "total_examples": len(rows), "output": str(output)}))


if __name__ == "__main__":
    main()
