"""Build an oracle-evidence extractive subset from official LongMemEval S.

Only benchmark turns marked has_answer=true are considered. A question is
eligible when its exact reference answer occurs in one such turn and survives
the same deterministic byte-context truncation used by the span evaluator.
This isolates span localization given gold evidence; it is not retrieval QA.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def _eligible_turn(question, answer, turns, block_size):
    """Choose the shortest gold evidence turn containing a visible answer span."""
    if not isinstance(question, str) or not isinstance(answer, str) or not answer:
        return None, "invalid_question_or_answer"
    question_suffix_bytes = len(f"\nQ:{question}\nA:".encode("utf-8"))
    memory_budget = block_size - 2 - question_suffix_bytes
    if memory_budget < 1:
        return None, "question_exceeds_context"
    candidates = []
    saw_evidence = False
    for turn in turns:
        if not isinstance(turn, dict) or turn.get("has_answer") is not True:
            continue
        saw_evidence = True
        content = turn.get("content")
        if not isinstance(content, str) or not content:
            continue
        # ByteTokenizer uses one token per UTF-8 byte. Match the evaluator's
        # head truncation so eligibility is independent of hidden gold offsets.
        fitted = content.encode("utf-8")[:memory_budget].decode("utf-8", errors="ignore")
        if answer in fitted:
            candidates.append(fitted)
    if candidates:
        # Shortest eligible evidence turn reduces unrelated text in the prompt.
        return min(candidates, key=len), None
    if not saw_evidence:
        return None, "no_labeled_evidence_turn"
    return None, "answer_not_literal_in_retained_evidence"


def build_cases(dataset, block_size=128):
    rows = []
    skipped = Counter()
    by_type = Counter()
    for item in dataset:
        qid = str(item.get("question_id", ""))
        qtype = str(item.get("question_type", "unknown"))
        if qid.endswith("_abs"):
            skipped["abstention"] += 1
            continue
        answer = item.get("answer")
        question = item.get("question")
        all_turns = []
        for session in item.get("haystack_sessions") or []:
            if isinstance(session, list):
                all_turns.extend(session)
        evidence, reason = _eligible_turn(question, answer, all_turns, block_size)
        if reason:
            skipped[reason] += 1
            continue
        rows.append({
            "id": qid,
            "fact_cluster_id": qid,
            "question": question,
            "memories": [evidence],
            "expected_answer": answer,
            "question_type": qtype,
        })
        by_type[qtype] += 1
    summary = {
        "protocol": "oracle-evidence span localization; use only turns labeled has_answer=true; exact reference answer must occur in the context-truncated turn; no retrieval is evaluated",
        "block_size_bytes": block_size,
        "total_questions": len(dataset),
        "eligible_questions": len(rows),
        "skipped_questions": sum(skipped.values()),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "eligible_by_question_type": dict(sorted(by_type.items())),
    }
    return rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--block-size", type=int, default=128)
    args = parser.parse_args()
    if args.block_size < 8:
        parser.error("block-size must be at least 8")
    dataset = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        parser.error("input dataset must be a JSON array")
    rows, summary = build_cases(dataset, args.block_size)
    if not rows:
        raise SystemExit("No extractive LongMemEval cases fit the configured context")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    Path(args.summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
