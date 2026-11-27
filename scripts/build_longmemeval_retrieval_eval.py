"""Build and score a transparent lexical session-retrieval benchmark on LongMemEval-S.

This evaluates retrieval only: it ranks complete conversation sessions for each
question and checks whether any benchmark-labeled evidence session is retrieved.
It does not evaluate answer generation or semantic equivalence.
"""
import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def terms(text):
    return [t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if len(t) > 1]


def _session_text(session):
    if not isinstance(session, list):
        return ""
    parts = []
    for turn in session:
        if isinstance(turn, dict):
            role = turn.get("role", "")
            content = turn.get("content", "")
            if isinstance(content, str) and content.strip():
                parts.append(f"{role}: {content}")
    return "\n".join(parts)


def build_retrieval_cases(dataset):
    cases, skipped = [], Counter()
    for item in dataset:
        qid = str(item.get("question_id", ""))
        question = item.get("question")
        sessions = item.get("haystack_sessions")
        session_ids = item.get("haystack_session_ids")
        evidence = item.get("answer_session_ids", item.get("evidence"))
        if not isinstance(question, str) or not question.strip():
            skipped["invalid_question"] += 1
            continue
        if not isinstance(sessions, list) or not isinstance(session_ids, list) or len(sessions) != len(session_ids):
            skipped["invalid_session_alignment"] += 1
            continue
        if not isinstance(evidence, list) or not evidence:
            skipped["missing_evidence_ids"] += 1
            continue
        docs = []
        for sid, session in zip(session_ids, sessions):
            text = _session_text(session)
            if text:
                docs.append({"session_id": str(sid), "text": text, "relevant": str(sid) in {str(x) for x in evidence}})
        if not docs:
            skipped["no_nonempty_sessions"] += 1
            continue
        if not any(doc["relevant"] for doc in docs):
            skipped["evidence_not_in_haystack"] += 1
            continue
        cases.append({"id": qid, "question": question, "evidence_session_ids": [str(x) for x in evidence], "sessions": docs,
                      "question_type": str(item.get("question_type", "unknown"))})
    return cases, {"total_questions": len(dataset), "eligible_questions": len(cases),
                   "skipped_questions": sum(skipped.values()), "skipped_by_reason": dict(sorted(skipped.items())),
                   "protocol": "rank full conversation sessions with transparent lexical Jaccard; relevant if session ID is in benchmark evidence list"}


def rank_sessions(question, sessions):
    q = set(terms(question))
    ranked = []
    for doc in sessions:
        d = set(terms(doc["text"]))
        score = len(q & d) / max(1, len(q | d))
        ranked.append((score, doc["session_id"], bool(doc["relevant"])))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked



def rank_sessions_bm25(question, sessions, k1=1.5, b=0.75):
    """Rank sessions with BM25 (Okapi); IDF is computed over this question's haystack."""
    query_terms = terms(question)
    tokenized = [terms(doc["text"]) for doc in sessions]
    doc_freq = Counter(term for doc_terms in tokenized for term in set(doc_terms))
    count = len(tokenized)
    avg_len = sum(map(len, tokenized)) / count if count else 0.0
    ranked = []
    for doc, doc_terms in zip(sessions, tokenized):
        frequencies = Counter(doc_terms)
        score = 0.0
        for term in set(query_terms):
            tf = frequencies[term]
            if not tf:
                continue
            idf = math.log(1 + (count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            norm = tf + k1 * (1 - b + b * len(doc_terms) / max(avg_len, 1e-12))
            score += idf * tf * (k1 + 1) / norm
        ranked.append((score, doc["session_id"], bool(doc["relevant"])))
    ranked.sort(key=lambda x: (-x[0], x[1]))
    return ranked

def evaluate(cases, ks=(1, 3, 5, 10), ranker=rank_sessions):
    ranks, per_type = [], {}
    for case in cases:
        ranked = ranker(case["question"], case["sessions"])
        first = next((i + 1 for i, row in enumerate(ranked) if row[2]), None)
        ranks.append(first)
        bucket = per_type.setdefault(case["question_type"], [])
        bucket.append(first)
    count = len(ranks)
    metrics = {"questions": count, "retrieval_coverage": sum(x is not None for x in ranks) / count if count else 0.0,
               "mrr": sum(1 / x for x in ranks if x is not None) / count if count else 0.0,
               "median_first_relevant_rank": sorted(x for x in ranks if x is not None)[(sum(x is not None for x in ranks)-1)//2] if any(x is not None for x in ranks) else None}
    for k in ks:
        metrics[f"recall_at_{k}"] = sum(x is not None and x <= k for x in ranks) / count if count else 0.0
    metrics["by_question_type"] = {
        key: {"questions": len(values), "recall_at_5": sum(x is not None and x <= 5 for x in values) / len(values),
              "mrr": sum(1 / x for x in values if x is not None) / len(values)}
        for key, values in sorted(per_type.items())
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-cases", required=True)
    parser.add_argument("--output-summary", required=True)
    args = parser.parse_args()
    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        parser.error("input dataset must be a JSON array")
    cases, build_summary = build_retrieval_cases(data)
    if not cases:
        raise SystemExit("No LongMemEval retrieval cases could be built")
    results = evaluate(cases)
    Path(args.output_cases).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_cases).write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in cases), encoding="utf-8")
    summary = {"build": build_summary, "retrieval_jaccard": results, "retrieval_bm25": evaluate(cases, ranker=rank_sessions_bm25),
               "warning": "Jaccard and BM25 are lexical baselines, not semantic retrievers. Ties are deterministically ordered by session ID; BM25 IDF is computed per question haystack."}
    Path(args.output_summary).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
