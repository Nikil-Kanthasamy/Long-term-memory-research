"""Compare tiny-LM answers with retrieved memory enabled vs disabled.

Uses a small JSONL evaluation set with question, answer, and memory fields.
Reports lexical answer overlap and retrieval hit rate; it does not claim
semantic correctness from automated string matching.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.store import MemoryStore
from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel


def terms(text):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(prediction, reference):
    pred, gold = terms(prediction), terms(reference)
    if not pred or not gold:
        return 0.0
    overlap = len(pred & gold)
    precision, recall = overlap / len(pred), overlap / len(gold)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def load_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    required = {"question", "answer", "memory"}
    for index, case in enumerate(cases, 1):
        if not required.issubset(case) or not all(isinstance(case[k], str) and case[k].strip() for k in required):
            raise ValueError(f"case {index} must contain non-empty question, answer, and memory strings")
    if not cases:
        raise ValueError("evaluation set is empty")
    return cases


def answer(model, tokenizer, prompt, device, max_new_tokens):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) > model.config.block_size:
        ids = ids[:, -model.config.block_size:]
    # top_k=1 makes generation greedy (deterministic).
    generated = model.generate(ids, max_new_tokens=max_new_tokens, temperature=1.0, top_k=1)
    return tokenizer.decode(generated[0, ids.size(1):].tolist()).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", default="examples/tiny_lm_memory_qa_eval.jsonl")
    parser.add_argument("--output", default="reports/tiny-lm-memory-qa-eval.json")
    parser.add_argument("--database", default="reports/tiny-lm-memory-qa-eval.sqlite3")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    cases = load_cases(args.cases)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    tokenizer = ByteTokenizer()
    store = MemoryStore(args.database)
    user_id = "tiny-lm-memory-eval"
    # Fresh isolated evaluation namespace; remove stale rows from prior runs.
    for item in store.active_memories(user_id):
        store.delete_memory(user_id, item["id"])
    store.add_memories_batch(user_id, [{"content": c["memory"], "category": "eval"} for c in cases])

    rows = []
    for case in cases:
        retrieved = store.search(user_id, case["question"], limit=5, mode="lexical")
        memory_prompt = (
            "Use the memory if it helps.\nMemory:\n"
            + "\n".join("- " + item["content"] for item in retrieved)
            + f"\nQuestion: {case['question']}\nAnswer:"
        )
        no_memory_prompt = f"Question: {case['question']}\nAnswer:"
        with_memory = answer(model, tokenizer, memory_prompt, device, args.max_new_tokens)
        without_memory = answer(model, tokenizer, no_memory_prompt, device, args.max_new_tokens)
        rows.append({
            "question": case["question"],
            "expected_answer": case["answer"],
            "retrieved_memory_count": len(retrieved),
            "gold_memory_retrieved": any(item["content"] == case["memory"] for item in retrieved),
            "with_memory": with_memory,
            "without_memory": without_memory,
            "with_memory_token_f1": token_f1(with_memory, case["answer"]),
            "without_memory_token_f1": token_f1(without_memory, case["answer"]),
        })
    n = len(rows)
    summary = {
        "checkpoint": str(args.checkpoint),
        "cases": n,
        "retrieval_hit_rate": sum(r["gold_memory_retrieved"] for r in rows) / n,
        "mean_token_f1_with_memory": sum(r["with_memory_token_f1"] for r in rows) / n,
        "mean_token_f1_without_memory": sum(r["without_memory_token_f1"] for r in rows) / n,
        "mean_paired_token_f1_delta": sum(r["with_memory_token_f1"] - r["without_memory_token_f1"] for r in rows) / n,
        "metric_note": "Whitespace/punctuation-insensitive token overlap only; human or semantic judging is still required.",
        "results": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))
    print(f"Saved detailed results to {output}")


if __name__ == "__main__":
    main()
