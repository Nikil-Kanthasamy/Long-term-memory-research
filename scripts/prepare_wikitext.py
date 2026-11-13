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
    provenance = (
        "Dataset: Salesforce/wikitext, configuration wikitext-2-raw-v1\\n"
        "Source: https://huggingface.co/datasets/Salesforce/wikitext\\n"
        "License stated by dataset card: Creative Commons Attribution-ShareAlike 4.0 (CC BY-SA 4.0)\\n"
        "Paper: Merity et al. (2016), Pointer Sentinel Mixture Models, https://arxiv.org/abs/1609.07843\\n"
        "Training uses train.txt; validation.txt is held out for model selection; test.txt is not used by the trainer.\\n"
    )
    (output / "PROVENANCE.txt").write_text(provenance, encoding="utf-8")

    for split in ("train", "validation", "test"):
        lines = [row["text"] for row in dataset[split] if row["text"].strip()]
        path = output / f"{split}.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"{split}: {len(lines)} non-empty lines -> {path}")


if __name__ == "__main__":
    main()
