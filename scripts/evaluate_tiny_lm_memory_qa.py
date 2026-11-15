""""Compare memory and no-memory answers from a locally trained TinyLM.

Cases are JSONL objects with question, memories (list), and expected_answer.
The script reports generation accuracy separately from evidence retrieval hits.
"""
import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.store import MemoryStore
from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def retrieval_contains_expected(retrieved, expected_answer):
    """Check whether retrieved evidence contains the normalized gold answer."""
    expected = normalize(expected_answer)
    if not expected:
        return False
    return any(expected in normalize(item.get("content", "")) for item in retrieved)


def load_cases(path):
    cases = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = json.loads(line)
        if not isinstance(item.get("question"), str) or not isinstance(item.get("expected_answer"), str):
            raise ValueError(f"line {number}: question and expected_answer must be strings")
        memories = item.get("memories", [])
        if not isinstance(memories, list) or not all(isinstance(x, str) and x.strip() for x in memories):
            raise ValueError(f"line {number}: memories must be a list of non-empty strings")
        cases.append(item)
    if not cases:
        raise ValueError("no evaluation cases found")
    return cases


def generate_answer(model, tokenizer, prompt, max_new_tokens, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) > model.config.block_size:
        ids = ids[:, -model.config.block_size:]
    with torch.no_grad():
        model.eval()
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.block_size:]
            logits, _ = model(context)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
    prompt_length = min(len(tokenizer.encode(prompt)), model.config.block_size)
    return tokenizer.decode(ids[0, prompt_length:].tolist()).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cases", required=True)
    parser.add_argument("--database", help="SQLite path; defaults to an isolated temporary database")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    tokenizer = ByteTokenizer()
    temporary = tempfile.TemporaryDirectory() if not args.database else None
    database = args.database or str(Path(temporary.name) / "memory.sqlite3")
    store = MemoryStore(database)
    cases = load_cases(args.cases)
    results = []
    for index, case in enumerate(cases):
        user_id = f"tiny-lm-eval-{index}"
        for memory in case.get("memories", []):
            store.add_memory(user_id, memory, category="evaluation", importance=1.0, confidence=1.0)
        retrieved = store.search(user_id, case["question"], limit=5, mode="lexical")
        memory_context = "\n".join(f"- {item['content']}" for item in retrieved) or "(no matching memories)"
        prompts = {
            "no_memory": f"You are a concise assistant.\nUser: {case['question']}\nAssistant:",
            "memory": (
                "Use the supplied memory to answer.\n"
                f"Memory:\n{memory_context}\nUser: {case['question']}\nAssistant:"
            ),
        }
        condition_results = {}
        for condition, prompt in prompts.items():
            torch.manual_seed(1000 + index)
            answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, device)
            expected = normalize(case["expected_answer"])
            observed = normalize(answer)
            matched = bool(expected) and expected in observed
            condition_results[condition] = {"answer": answer, "matched": matched}
        results.append({
            "case_id": case.get("id", str(index)),
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "retrieved_memory_count": len(retrieved),
            "retrieval_hit": retrieval_contains_expected(retrieved, case["expected_answer"]),
            **condition_results,
        })

    summary = {}
    for condition in ("no_memory", "memory"):
        hits = sum(row[condition]["matched"] for row in results)
        summary[condition] = {"correct": hits, "total": len(results), "exact_contains_accuracy": hits / len(results)}
    retrieval_hits = sum(row["retrieval_hit"] for row in results)
    report = {
        "evaluation": "synthetic_memory_qa_with_retrieval_evidence",
        "checkpoint_stage": checkpoint.get("stage", "base_language_model"),
        "checkpoint_step": checkpoint.get("step"),
        "cases": len(results),
        "summary": summary,
        "retrieval_summary": {
            "gold_answer_present_in_retrieved_evidence": retrieval_hits,
            "total": len(results),
            "hit_rate": retrieval_hits / len(results),
        },
        "results": results,
        "limitations": [
            "Synthetic tiny evaluation set; not a benchmark of general assistant ability.",
            "Generation uses literal normalized answer-substring scoring; paraphrases may be marked wrong.",
            "Retrieval hit checks literal gold-answer presence in retrieved text, not semantic evidence quality.",
            "A model that generates gibberish may score zero even when retrieval succeeds.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "retrieval_summary": report["retrieval_summary"], "output": str(output)}, indent=2))
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
