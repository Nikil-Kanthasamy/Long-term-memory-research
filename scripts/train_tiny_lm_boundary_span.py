"""Train answer start/end pointers without supplying the gold answer length.

This is a constrained extractive boundary-localization experiment. The model
predicts start and end byte positions independently from the question state;
evaluation never uses answer length to construct candidate spans.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel
from scripts.train_tiny_lm_span_copy import PROMPT_RE, fit_memory_prompt, load_jsonl, locate_answer_span


class SpanBoundaryHead(nn.Module):
    """Predict start and end positions using separate query projections."""

    def __init__(self, d_model):
        super().__init__()
        self.start_query = nn.Linear(d_model, d_model, bias=False)
        self.end_query = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_hidden, memory_keys):
        scale = memory_keys.size(-1) ** 0.5
        start_logits = torch.matmul(self.start_query(query_hidden), memory_keys.transpose(1, 2)) / scale
        end_logits = torch.matmul(self.end_query(query_hidden), memory_keys.transpose(1, 2)) / scale
        return start_logits, end_logits


def prepare_examples(rows, block_size, tokenizer=None):
    tokenizer = tokenizer or ByteTokenizer()
    prepared = []
    skipped_missing = 0
    skipped_long = 0
    for row in rows:
        try:
            ids, start, end, query_position, memory_end = locate_answer_span(
                row["prompt"], row["response"].rstrip(), tokenizer
            )
        except ValueError as error:
            if "answer span is not present" not in str(error):
                raise
            skipped_missing += 1
            continue
        if len(ids) > block_size:
            skipped_long += 1
            continue
        prepared.append((ids, start, end, query_position, memory_end))
    return prepared, skipped_missing, skipped_long


def boundary_loss(start_logits, end_logits, start_target, end_target):
    return F.cross_entropy(start_logits, start_target) + F.cross_entropy(end_logits, end_target)


def score_boundary_span(start_logits, end_logits, start, end):
    """Rank true boundaries among memory positions; no answer length is used."""
    start_values = start_logits.detach().float().cpu().flatten()
    end_values = end_logits.detach().float().cpu().flatten()
    true_start = start
    true_end = end - 1
    start_order = torch.argsort(start_values, descending=True)
    end_order = torch.argsort(end_values, descending=True)
    start_rank = int((start_order == true_start).nonzero(as_tuple=True)[0].item()) + 1
    end_rank = int((end_order == true_end).nonzero(as_tuple=True)[0].item()) + 1
    predicted_start = int(start_values.argmax().item())
    predicted_end = int(end_values.argmax().item())
    return {
        "start_rank": start_rank,
        "end_rank": end_rank,
        "start_match": predicted_start == true_start,
        "end_match": predicted_end == true_end,
        "exact_span_match": predicted_start == true_start and predicted_end == true_end,
        "predicted_valid": predicted_start <= predicted_end,
        "predicted_start": predicted_start,
        "predicted_end_inclusive": predicted_end,
    }


def _boundary_logits(model, head, ids, query_position, memory_end, device):
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, hidden = model(input_ids, return_hidden=True)
    query_hidden = hidden[:, query_position:query_position + 1, :]
    memory_keys = model.token_embedding(input_ids[:, 2:memory_end])
    start_logits, end_logits = head(query_hidden, memory_keys)
    return start_logits.squeeze(1), end_logits.squeeze(1)


def evaluate(model, head, cases, device, block_size):
    tokenizer = ByteTokenizer()
    model.eval()
    head.eval()
    results = []
    skipped_missing = 0
    skipped_outside = 0
    with torch.no_grad():
        for case in cases:
            memories = case.get("memories") or []
            if not memories:
                raise ValueError("each evaluation case needs at least one memory")
            prompt = fit_memory_prompt(case["question"], memories[0], block_size, tokenizer)
            try:
                ids, start, end, query_position, memory_end = locate_answer_span(
                    prompt, case["expected_answer"], tokenizer
                )
            except ValueError as error:
                if "answer span is not present" not in str(error):
                    raise
                skipped_missing += 1
                skipped_outside += 1
                continue
            start_logits, end_logits = _boundary_logits(
                model, head, ids, query_position, memory_end, device
            )
            scores = score_boundary_span(start_logits, end_logits, start - 2, end - 2)
            scores["case_id"] = case.get("id")
            scores["memory_positions"] = memory_end - 2
            results.append(scores)
    if not results:
        raise ValueError("no evaluation cases contain an answer span verbatim in retained memory")
    count = len(results)
    return {
        "examples": count,
        "skipped_missing_answer_span": skipped_missing,
        "skipped_answer_outside_answer_independent_context": skipped_outside,
        "start_top1_accuracy": sum(row["start_match"] for row in results) / count,
        "end_top1_accuracy": sum(row["end_match"] for row in results) / count,
        "exact_span_accuracy": sum(row["exact_span_match"] for row in results) / count,
        "valid_predicted_span_rate": sum(row["predicted_valid"] for row in results) / count,
        "mean_start_rank": sum(row["start_rank"] for row in results) / count,
        "mean_end_rank": sum(row["end_rank"] for row in results) / count,
        "context_protocol": "answer-independent deterministic head truncation",
        "candidate_protocol": "independent start/end classification over memory bytes; no gold answer length or length-conditioned windows",
        "per_case": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True, help="Copy-enabled base checkpoint")
    parser.add_argument("--output-checkpoint", default="models/tiny-memory-lm-copy/boundary-span.pt")
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--output", default="copy-boundary-span-evaluation.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=43)
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
    head = SpanBoundaryHead(model.config.d_model).to(device)
    rows = load_jsonl(args.input)
    prepared, skipped_missing, skipped_long = prepare_examples(rows, model.config.block_size)
    print(f"Prepared {len(prepared)} boundary examples; skipped {skipped_missing} non-extractive and {skipped_long} over-context rows.")
    if not prepared:
        parser.error("no extractive training examples fit the context")
    eval_cases = load_jsonl(args.eval_cases)
    baseline = evaluate(model, head, eval_cases, device, model.config.block_size)
    print("Before boundary training: " + json.dumps({k: v for k, v in baseline.items() if k != "per_case"}))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.learning_rate)
    model.train()
    head.train()
    for step in range(1, args.steps + 1):
        batch = random.choices(prepared, k=min(args.batch_size, len(prepared)))
        losses = []
        for ids, start, end, query_position, memory_end in batch:
            start_logits, end_logits = _boundary_logits(
                model, head, ids, query_position, memory_end, device
            )
            losses.append(boundary_loss(
                start_logits, end_logits,
                torch.tensor([start - 2], dtype=torch.long, device=device),
                torch.tensor([end - 3], dtype=torch.long, device=device),
            ))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step={step} boundary_loss={loss.item():.4f}")
    trained = evaluate(model, head, eval_cases, device, model.config.block_size)
    payload = {
        "model": model.checkpoint(),
        "span_boundary_head": head.state_dict(),
        "stage": "answer_length_free_boundary_supervision",
        "steps": args.steps,
        "training_examples": len(prepared),
        "objective": "sum_of_start_and_end_boundary_cross_entropy",
    }
    output_checkpoint = Path(args.output_checkpoint)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_checkpoint)
    report = {"baseline_before_supervision": baseline, "after_boundary_supervision": trained}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("After boundary training: " + json.dumps({k: v for k, v in trained.items() if k != "per_case"}, indent=2))
    print(f"Saved boundary-supervised checkpoint to {output_checkpoint}")


if __name__ == "__main__":
    main()
