"""Prepare the public WikiText-2 raw split as plain-text files.

Install the optional dataset dependency with:
    python -m pip install -r requirements-tiny-lm-data.txt
"""
import argparse
from pathlib import Path

from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/wikitext-2")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    for split in ("train", "validation", "test"):
        lines = [row["text"] for row in dataset[split] if row["text"].strip()]
        path = output / f"{split}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(lines)} non-empty lines -> {path}")


if __name__ == "__main__":
    main()
