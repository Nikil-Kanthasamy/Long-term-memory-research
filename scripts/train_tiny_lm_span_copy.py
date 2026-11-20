"""Explicitly supervise pointer attention to select answer spans in retrieved memory.

This is a diagnostic extractive task, not free-form answer generation. Training
uses answer offsets in the memory passage; evaluation reports attention mass
and rank of the gold span among memory windows of the same byte length.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel


PROMPT_RE = re.compile(r"M:(.*?)\nQ:(.*?)\nA:", re.DOTALL)


def locate_answer_span(prompt, answer, tokenizer=None):
    """Return prompt IDs and gold answer span offsets inside the memory prefix."""
    tokenizer = tokenizer or ByteTokenizer()
    match = PROMPT_RE.fullmatch(prompt)
    if not match:
        raise ValueError("prompt must use the canonical M:/Q:/A: format")
    memory, question = match.groups()
    memory_bytes = tokenizer.encode(memory)
    answer_bytes = tokenizer.encode(answer)
    if not answer_bytes:
        raise ValueError("answer cannot be empty")
    start_in_memory = bytes(memory_bytes).find(bytes(answer_bytes))
    if start_in_memory < 0:
        raise ValueError("answer span is not present in the retrieved memory")
    prompt_ids = tokenizer.encode(f"M:{memory}\nQ:{question}\nA:")
    start = 2 + start_in_memory
    end = start + len(answer_bytes)
    if end > len(prompt_ids):
        raise ValueError("answer span falls outside the memory prefix")
    return prompt_ids, start, end, len(prompt_ids) - 1, 2 + len(memory_bytes)


def score_answer_span(attention, start, end, memory_start, memory_end):
    """Score gold-span attention mass and its rank among equal-length memory spans."""
    values = attention.detach().float().cpu().tolist() if hasattr(attention, "detach") else list(attention)
    if not (0 <= memory_start <= start < end <= memory_end <= len(values)):
        raise ValueError("answer span must lie inside the memory region")
    gold_score = sum(values[start:end])
    width = end - start
    candidates = [
        (sum(values[index:index + width]), index)
        for index in range(memory_start, memory_end - width + 1)
    ]
    rank = 1 + sum(score > gold_score + 1e-12 for score, _ in candidates)
    best_score, best_start = max(candidates, default=(0.0, -1))
    return {
        "gold_span_mass": gold_score,
        "gold_span_rank": rank,
        "candidate_spans": len(candidates),
        "top_span_match": best_start == start,
        "top_span_mass": best_score,
    }


def load_jsonl(path):
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"line {line_number}: expected an object")
        rows.append(row)
    if not rows:
        raise ValueError("dataset contains no rows")
    return rows


def evaluate(model, cases, device, block_size):
    tokenizer = ByteTokenizer()
    results = []
    model.eval()
    with torch.no_grad():
        for case in cases:
            memories = case.get("memories") or []
            if not memories:
                raise ValueError("each evaluation case needs at least one memory")
            prompt = f"M:{memories[0]}\nQ:{case['question']}\nA:"
            ids, start, end, query_position, memory_end = locate_answer_span(
                prompt, case["expected_answer"], tokenizer
            )
            if len(ids) > block_size:
                raise ValueError("evaluation prompt exceeds model block_size")
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _, _, attention = model(input_ids, return_copy_attention=True)
            scores = score_answer_span(
                attention[0, query_position], start, end, 2, memory_end
            )
            scores["case_id"] = case.get("id")
            results.append(scores)
    count = len(results)
    return {
        "examples": count,
        "mean_gold_span_mass": sum(row["gold_span_mass"] for row in results) / count,
        "top_span_accuracy": sum(row["top_span_match"] for row in results) / count,
        "mean_gold_span_rank": sum(row["gold_span_rank"] for row in results) / count,
        "mean_candidate_spans": sum(row["candidate_spans"] for row in results) / count,
        "rank_protocol": "gold byte length is supplied to define candidate windows; this is a span-localization diagnostic, not answer-length-free extraction",
        "per_case": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="JSONL prompt/response examples")
    parser.add_argument("--checkpoint", required=True, help="Copy-enabled base checkpoint")
    parser.add_argument("--output-checkpoint", default="models/tiny-memory-lm-copy/span-copy.pt")
    parser.add_argument("--eval-cases", help="Optional JSONL held-out QA cases")
    parser.add_argument("--output", default="span-copy-evaluation.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.batch_size < 1:
        parser.error("steps and batch-size must be positive")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    if not model.config.copy_enabled:
        parser.error("checkpoint must have the causal copy mechanism enabled")
    tokenizer = ByteTokenizer()
    rows = load_jsonl(args.input)
    prepared = []
    for row in rows:
        ids, start, end, query_position, memory_end = locate_answer_span(
            row["prompt"], row["response"].rstrip()
        )
        if len(ids) > model.config.block_size:
            continue
        prepared.append((ids, start, end, query_position))
    if not prepared:
        parser.error("no training examples fit the model context")
    eval_cases = load_jsonl(args.eval_cases) if args.eval_cases else None
    baseline_metrics = evaluate(model, eval_cases, device, model.config.block_size) if eval_cases else None
    if baseline_metrics:
        print("Before span supervision: " + json.dumps({key: value for key, value in baseline_metrics.items() if key != "per_case"}))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    for step in range(1, args.steps + 1):
        batch = random.choices(prepared, k=min(args.batch_size, len(prepared)))
        losses = []
        for ids, start, end, query_position in batch:
            input_ids = torch.tensor([ids], dtype=torch.long, device=device)
            _, _, attention = model(input_ids, return_copy_attention=True)
            gold_mass = attention[0, query_position, start:end].sum()
            losses.append(-torch.log(gold_mass.clamp_min(1e-9)))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step={step} answer_span_loss={loss.item():.4f}")
    payload = model.checkpoint()
    payload["tokenizer"] = checkpoint.get("tokenizer", "utf8-byte-v1")
    payload.update({
        "stage": "supervised_memory_span_selection",
        "span_steps": args.steps,
        "span_training_examples": len(prepared),
        "objective": "negative_log_probability_mass_on_gold_memory_answer_span",
    })
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_checkpoint)
    print(f"Saved span-supervised checkpoint to {output_checkpoint}")
    if eval_cases:
        trained_metrics = evaluate(model, eval_cases, device, model.config.block_size)
        report = {"baseline_before_supervision": baseline_metrics, "after_span_supervision": trained_metrics}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\\n", encoding="utf-8")
        print("After span supervision: " + json.dumps({key: value for key, value in trained_metrics.items() if key != "per_case"}, indent=2))


if __name__ == "__main__":
    main()
