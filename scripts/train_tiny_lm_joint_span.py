"""Train a joint, answer-length-free start/end span selector."""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel
from scripts.train_tiny_lm_boundary_span import prepare_examples
from scripts.train_tiny_lm_span_copy import load_jsonl, fit_memory_prompt, locate_answer_span


class JointSpanHead(nn.Module):
    """Score ordered (start, end) pairs jointly, without a gold-length input."""

    def __init__(self, d_model):
        super().__init__()
        self.start_query = nn.Linear(d_model, d_model, bias=False)
        self.end_query = nn.Linear(d_model, d_model, bias=False)
        self.start_key = nn.Linear(d_model, d_model, bias=False)
        self.end_key = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query_hidden, memory_keys):
        scale = memory_keys.size(-1) ** 0.5
        start_q = self.start_query(query_hidden)
        end_q = self.end_query(query_hidden)
        start_k = self.start_key(memory_keys)
        end_k = self.end_key(memory_keys)
        start_logits = torch.matmul(start_q, memory_keys.transpose(1, 2)) / scale
        end_logits = torch.matmul(end_q, memory_keys.transpose(1, 2)) / scale
        start_pair = start_k * start_q
        end_pair = end_k * end_q
        compatibility = torch.matmul(start_pair, end_pair.transpose(1, 2)) / scale
        start_logits = start_logits.squeeze(1)
        end_logits = end_logits.squeeze(1)
        pair_logits = start_logits.unsqueeze(-1) + end_logits.unsqueeze(-2) + compatibility
        return start_logits, end_logits, pair_logits


def mask_invalid_spans(pair_logits, max_span_bytes=None):
    """Keep only start <= end pairs; optional cap is fixed, never gold-conditioned."""
    length = pair_logits.size(-1)
    starts = torch.arange(length, device=pair_logits.device).unsqueeze(1)
    ends = torch.arange(length, device=pair_logits.device).unsqueeze(0)
    valid = ends >= starts
    if max_span_bytes is not None:
        valid = valid & ((ends - starts + 1) <= max_span_bytes)
    return pair_logits.masked_fill(~valid, torch.finfo(pair_logits.dtype).min)


def joint_span_loss(pair_logits, start_target, end_target):
    length = pair_logits.size(-1)
    target = start_target * length + end_target
    return F.cross_entropy(pair_logits.flatten(1), target)


def score_joint_span(pair_logits, start, end, span_width_penalty=0.0):
    """Rank spans after an optional fixed per-byte penalty for wider spans."""
    if span_width_penalty < 0:
        raise ValueError("span_width_penalty must be non-negative")
    adjusted = mask_invalid_spans(pair_logits).detach().float()
    length = adjusted.size(-1)
    starts = torch.arange(length, device=adjusted.device).unsqueeze(1)
    ends = torch.arange(length, device=adjusted.device).unsqueeze(0)
    widths = (ends - starts + 1).clamp_min(1)
    adjusted = adjusted - span_width_penalty * widths
    scores = adjusted.flatten()
    target = start * length + end
    order = torch.argsort(scores, descending=True)
    rank = int((order == target).nonzero(as_tuple=True)[0].item()) + 1
    predicted = int(scores.argmax().item())
    length = pair_logits.size(-1)
    pred_start, pred_end = divmod(predicted, length)
    return {
        "joint_rank": rank,
        "exact_span_match": pred_start == start and pred_end == end,
        "predicted_start": pred_start,
        "predicted_end_inclusive": pred_end,
    }


def _logits(model, head, ids, query_position, memory_end, device):
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    _, _, hidden = model(input_ids, return_hidden=True)
    query = hidden[:, query_position:query_position + 1, :]
    keys = model.token_embedding(input_ids[:, 2:memory_end])
    start, end, pair = head(query, keys)
    return start, end, pair


def evaluate(model, head, cases, device, block_size):
    tokenizer = ByteTokenizer()
    model.eval()
    head.eval()
    results = []
    skipped = 0
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
                skipped += 1
                continue
            start_logits, end_logits, pair_logits = _logits(
                model, head, ids, query_position, memory_end, device
            )
            # locate_answer_span uses exclusive end; memory keys start at prompt offset 2.
            true_start, true_end = start - 2, end - 3
            row = score_joint_span(pair_logits, true_start, true_end)
            row["start_top1"] = int(start_logits.argmax().item()) == true_start
            row["end_top1"] = int(end_logits.argmax().item()) == true_end
            row["case_id"] = case.get("id")
            results.append(row)
    if not results:
        raise ValueError("no evaluation cases contain an answer span in retained memory")
    count = len(results)
    return {
        "examples": count,
        "skipped_missing_answer_span": skipped,
        "start_top1_accuracy": sum(x["start_top1"] for x in results) / count,
        "end_top1_accuracy": sum(x["end_top1"] for x in results) / count,
        "joint_exact_span_accuracy": sum(x["exact_span_match"] for x in results) / count,
        "mean_joint_span_rank": sum(x["joint_rank"] for x in results) / count,
        "candidate_protocol": "all ordered spans start<=end over retained memory bytes; no gold answer length or length-conditioned windows",
        "per_case": results,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-checkpoint", default="models/tiny-memory-lm-copy/joint-span.pt")
    parser.add_argument("--eval-cases", required=True)
    parser.add_argument("--output", default="copy-joint-span-evaluation.json")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=47)
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
    head = JointSpanHead(model.config.d_model).to(device)
    rows = load_jsonl(args.input)
    prepared, skipped_missing, skipped_long = prepare_examples(rows, model.config.block_size)
    print(f"Prepared {len(prepared)} joint-span examples; skipped {skipped_missing} non-extractive and {skipped_long} over-context rows.")
    if not prepared:
        parser.error("no extractive training examples fit the context")
    eval_cases = load_jsonl(args.eval_cases)
    baseline = evaluate(model, head, eval_cases, device, model.config.block_size)
    print("Before joint-span training: " + json.dumps({k: v for k, v in baseline.items() if k != "per_case"}))
    params = list(model.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    model.train()
    head.train()
    for step in range(1, args.steps + 1):
        batch = random.choices(prepared, k=min(args.batch_size, len(prepared)))
        losses = []
        for ids, start, end, query_position, memory_end in batch:
            _, _, pair = _logits(model, head, ids, query_position, memory_end, device)
            pair = mask_invalid_spans(pair)
            losses.append(joint_span_loss(
                pair,
                torch.tensor([start - 2], dtype=torch.long, device=device),
                torch.tensor([end - 3], dtype=torch.long, device=device),
            ))
        loss = torch.stack(losses).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step == 1 or step % max(1, args.steps // 10) == 0 or step == args.steps:
            print(f"step={step} joint_span_loss={loss.item():.4f}")
    trained = evaluate(model, head, eval_cases, device, model.config.block_size)
    payload = {
        "model": model.checkpoint(),
        "joint_span_head": head.state_dict(),
        "stage": "answer_length_free_joint_span_supervision",
        "steps": args.steps,
        "training_examples": len(prepared),
        "objective": "cross_entropy_over_all_valid_ordered_start_end_pairs",
    }
    output = Path(args.output_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    Path(args.output).write_text(json.dumps({
        "baseline_before_supervision": baseline,
        "after_joint_span_supervision": trained,
    }, indent=2) + "\n", encoding="utf-8")
    print("After joint-span training: " + json.dumps({k: v for k, v in trained.items() if k != "per_case"}, indent=2))
    print(f"Saved joint-span checkpoint to {output}")


if __name__ == "__main__":
    main()
