"""Build and score turn-level evidence retrieval on official LongMemEval-S.

Only turns explicitly labeled has_answer=true are relevant. This evaluates
retrieval of annotated evidence turns, not answer generation or semantic
answer correctness. Abstention questions and coverage failures are reported.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def terms(text):
    return [term for term in re.findall(r"[a-z0-9]+", str(text).lower()) if len(term) > 1]


def build_turn_cases(dataset):
    cases, skipped = [], Counter()
    abstentions = 0
    for item in dataset:
        qid = str(item.get("question_id", ""))
        question = item.get("question")
        if qid.endswith("_abs"):
            abstentions += 1
            continue
        if not isinstance(question, str) or not question.strip():
            skipped["invalid_question"] += 1
            continue
        sessions = item.get("haystack_sessions")
        session_ids = item.get("haystack_session_ids")
        if not isinstance(sessions, list) or not isinstance(session_ids, list) or len(sessions) != len(session_ids):
            skipped["invalid_session_alignment"] += 1
            continue
        turns = []
        for session_index, (session_id, session) in enumerate(zip(session_ids, sessions)):
            if not isinstance(session, list):
                continue
            for turn_index, turn in enumerate(session):
                if not isinstance(turn, dict):
                    continue
                content = turn.get("content")
                if not isinstance(content, str) or not content.strip():
                    continue
                turns.append({
                    "turn_id": f"{session_index}:{turn_index}",
                    "session_id": str(session_id),
                    "session_index": session_index,
                    "turn_index": turn_index,
                    "role": str(turn.get("role", "unknown")),
                    "date": str((item.get("haystack_dates") or [])[session_index])
                            if session_index < len(item.get("haystack_dates") or []) else "",
                    "text": content.strip(),
                    "relevant": turn.get("has_answer") is True,
                })
        gold_count = sum(turn["relevant"] for turn in turns)
        if not turns:
            skipped["no_nonempty_turns"] += 1
            continue
        if not gold_count:
            skipped["no_labeled_evidence_turn"] += 1
            continue
        cases.append({
            "id": qid,
            "question": question,
            "question_type": str(item.get("question_type", "unknown")),
            "gold_turn_count": gold_count,
            "turns": turns,
        })
    summary = {
        "protocol": "rank individual nonempty conversation turns; relevant iff official turn label has_answer=true; no generation or semantic judging",
        "total_questions": len(dataset),
        "answerable_questions": len(dataset) - abstentions,
        "abstention_questions": abstentions,
        "eligible_questions": len(cases),
        "skipped_questions": sum(skipped.values()),
        "skipped_by_reason": dict(sorted(skipped.items())),
        "turns_total": sum(len(case["turns"]) for case in cases),
        "gold_evidence_turns": sum(case["gold_turn_count"] for case in cases),
        "coverage_of_answerable_questions": len(cases) / (len(dataset) - abstentions) if len(dataset) > abstentions else 0.0,
    }
    return cases, summary


def rank_jaccard(question, turns):
    query = set(terms(question))
    ranked = []
    for turn in turns:
        doc = set(terms(turn["text"]))
        score = len(query & doc) / max(1, len(query | doc))
        ranked.append((score, turn["turn_id"], bool(turn["relevant"])))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked


def rank_bm25(question, turns, k1=1.5, b=0.75):
    query_terms = set(terms(question))
    tokenized = [terms(turn["text"]) for turn in turns]
    count = len(tokenized)
    average_length = sum(map(len, tokenized)) / count if count else 0.0
    document_frequency = Counter(term for tokens in tokenized for term in set(tokens))
    ranked = []
    for turn, tokens in zip(turns, tokenized):
        frequencies = Counter(tokens)
        score = 0.0
        for term in query_terms:
            tf = frequencies[term]
            if not tf:
                continue
            df = document_frequency[term]
            inverse_document_frequency = math.log(1.0 + (count - df + 0.5) / (df + 0.5))
            denominator = tf + k1 * (1.0 - b + b * len(tokens) / max(average_length, 1e-12))
            score += inverse_document_frequency * tf * (k1 + 1.0) / denominator
        ranked.append((score, turn["turn_id"], bool(turn["relevant"])))
    ranked.sort(key=lambda row: (-row[0], row[1]))
    return ranked


def evaluate_turn_retrieval(cases, ranker, ks=(1, 3, 5, 10)):
    per_question = []
    by_type = {}
    for case in cases:
        ranked = ranker(case["question"], case["turns"])
        gold_count = case["gold_turn_count"]
        first = next((index + 1 for index, row in enumerate(ranked) if row[2]), None)
        row = {
            "question_id": case["id"],
            "question_type": case["question_type"],
            "gold_turn_count": gold_count,
            "first_relevant_rank": first,
            "mrr": 1.0 / first if first else 0.0,
            "top_results": [
                {"turn_id": turn_id, "relevant": relevant, "score": score}
                for score, turn_id, relevant in ranked[:max(ks)]
            ],
        }
        for k in ks:
            hits = sum(bool(entry[2]) for entry in ranked[:k])
            row[f"precision_at_{k}"] = hits / k
            row[f"recall_at_{k}"] = hits / gold_count
            row[f"hit_at_{k}"] = float(hits > 0)
        per_question.append(row)
        by_type.setdefault(case["question_type"], []).append(row)
    count = len(per_question)
    metrics = {
        "questions": count,
        "mrr": sum(row["mrr"] for row in per_question) / count if count else 0.0,
        "by_question_type": {},
    }
    for k in ks:
        for metric in ("precision", "recall", "hit"):
            key = f"{metric}_at_{k}"
            metrics[key] = sum(row[key] for row in per_question) / count if count else 0.0
    for qtype, rows in sorted(by_type.items()):
        bucket = {"questions": len(rows), "mrr": sum(row["mrr"] for row in rows) / len(rows)}
        for k in ks:
            for metric in ("precision", "recall", "hit"):
                key = f"{metric}_at_{k}"
                bucket[key] = sum(row[key] for row in rows) / len(rows)
        metrics["by_question_type"][qtype] = bucket
    return {"metrics": metrics, "per_question": per_question}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-summary", required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    if args.k < 1:
        parser.error("k must be at least 1")
    dataset = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(dataset, list):
        parser.error("input dataset must be a JSON array")
    cases, build_summary = build_turn_cases(dataset)
    if not cases:
        raise SystemExit("No LongMemEval turn-level evidence cases could be built")
    ks = tuple(sorted(set((1, 3, 5, args.k))))
    jaccard = evaluate_turn_retrieval(cases, rank_jaccard, ks)
    bm25 = evaluate_turn_retrieval(cases, rank_bm25, ks)
    Path(args.output_cases).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_cases).write_text(
        "".join(json.dumps(case, ensure_ascii=False) + "\n" for case in cases),
        encoding="utf-8",
    )
    summary = {
        "build": build_summary,
        "retrieval_jaccard": jaccard,
        "retrieval_bm25": bm25,
        "warning": "Turn-level evidence retrieval only. Labels are benchmark has_answer flags; this is not answer generation or semantic correctness. Metrics are averaged over eligible answerable questions.",
    }
    Path(args.output_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"build": build_summary, "retrieval_jaccard": jaccard["metrics"],
                      "retrieval_bm25": bm25["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
