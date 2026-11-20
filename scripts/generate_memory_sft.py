"""Build a deterministic, varied memory-grounding SFT curriculum.

The synthetic training facts are deliberately separate from the held-out QA
fixtures. Each fact is expressed through multiple memory phrasings and
question phrasings so the model cannot rely on one fixed sentence template.
"""
import argparse
import json
from pathlib import Path

FACTS = [
    ("cache lifetime", "17 minutes"),
    ("export format", "CBOR"),
    ("replica region", "ap-south-2"),
    ("retry limit", "7 attempts"),
    ("workspace code", "ORBIT-731"),
    ("snapshot cadence", "every 23 hours"),
    ("queue name", "copper-fern"),
    ("shard count", "13 shards"),
    ("checksum method", "BLAKE2b"),
    ("archive tier", "cold-glacier"),
    ("lease duration", "41 seconds"),
    ("schema revision", "rev-208"),
    ("compression mode", "zstandard"),
    ("worker pool", "willow-pool"),
    ("page size", "37 records"),
    ("rotation window", "11 days"),
]

MEMORY_TEMPLATES = (
    "The {field} is {value}.",
    "Saved setting — {field}: {value}.",
    "Memory record: {field} = {value}.",
)

QUESTION_TEMPLATES = (
    "What is the {field}?",
    "Which value is set for {field}?",
    "State the {field} from memory.",
    "What does the saved record list as the {field}?",
    "Can you retrieve the {field}?",
    "According to the note, what is the {field}?",
)


def build_examples():
    """Return 288 deterministic examples with 3 × 6 surface-form variation."""
    rows = []
    for fact_index, (field, value) in enumerate(FACTS):
        for memory_index, memory_template in enumerate(MEMORY_TEMPLATES):
            memory = memory_template.format(field=field, value=value)
            for question_index, question_template in enumerate(QUESTION_TEMPLATES):
                question = question_template.format(field=field)
                rows.append({
                    "prompt": f"M:{memory}\nQ:{question}\nA:",
                    "response": value,
                    "id": f"grounding-{fact_index:02d}-{memory_index:02d}-{question_index:02d}",
                })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="examples/tiny_lm_sft_demo.jsonl")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_path = Path(args.base)
    base = [
        json.loads(line)
        for line in base_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = base + build_examples()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(json.dumps({
        "base_examples": len(base),
        "generated_grounding_examples": len(rows) - len(base),
        "total_examples": len(rows),
        "unique_fact_values": len(FACTS),
        "memory_phrasings_per_fact": len(MEMORY_TEMPLATES),
        "question_phrasings_per_fact": len(QUESTION_TEMPLATES),
        "output": str(output),
    }))


if __name__ == "__main__":
    main()
