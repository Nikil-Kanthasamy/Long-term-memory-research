"""Download the official LongMemEval JSON without adding it to Git."""
import argparse
from pathlib import Path
from urllib.request import urlretrieve

FILES = {
    "s": "longmemeval_s_cleaned.json",
    "oracle": "longmemeval_oracle.json",
    "m": "longmemeval_m_cleaned.json",
}
BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/"


def main():
    parser = argparse.ArgumentParser(description="Download official LongMemEval data")
    parser.add_argument("--variant", choices=FILES, default="s")
    parser.add_argument("--output-dir", default="data")
    args = parser.parse_args()
    filename = FILES[args.variant]
    destination = Path(args.output_dir) / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise SystemExit(f"{destination} already exists; move it or delete it before downloading again")
    url = BASE + filename
    print(f"Downloading {url}")
    urlretrieve(url, destination)
    print(f"Saved dataset to {destination}")
    print("Dataset files are ignored by Git via the data/ entry in .gitignore.")


if __name__ == "__main__":
    main()
