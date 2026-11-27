"""Session-level BM25 evaluation on PRAGMA's personalized-guidance benchmark.

The released PRAGMA data is under the NVIDIA License for non-commercial
research/evaluation. Download it at run time; never commit source conversations.
Only aggregate and per-question retrieval metrics are written to the artifact.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

try:
    from scripts.build_longmemeval_turn_retrieval_eval import (
        BM25_ABLATION_STOPWORDS,
        BM25_SELECTIVE_STOPWORDS,
        evaluate_turn_retrieval,
        rank_bm25,
        rank_jaccard,
        rank_rrf,
    )
except ModuleNotFoundError:
    from build_longmemeval_turn_retrieval_eval import (
        BM25_ABLATION_STOPWORDS,
        BM25_SELECTIVE_STOPWORDS,
        evaluate_turn_retrieval,
        rank_bm25,
        rank_jaccard,
    )


QUERY_TYPES = {
    "type1": "event-align",
    "type2": "event-correct",
    "type3": "trajectory-align",
    "type4": "trajectory-correct",
}


def _parse_time(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing timestamp: {value!r}")
    parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def _session_text(session):
    if isinstance(session, str):
        return session.strip()
    if not isinstance(session, list):
        return ""
    parts = []
    for message in session:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        role = str(message.get("role", "unknown")).strip()
        parts.append(f"{role}: {content.strip()}")
    return "\n".join(parts)


def build_pragma_cases(full_sessions, metadata):
    """Join users, flatten each session to a retrieval unit, enforce time cutoff."""
    if not isinstance(full_sessions, list) or not isinstance(metadata, list):
        raise ValueError("PRAGMA full_sessions and metadata must both be JSON arrays")
    users = {}
    for user in full_sessions:
        if not isinstance(user, dict) or not user.get("user_id"):
            raise ValueError("Malformed PRAGMA user record")
        uid = str(user["user_id"])
        if uid in users:
            raise ValueError(f"Duplicate PRAGMA user_id: {uid}")
        users[uid] = user

    cases = []
    excluded = {
        "unknown_user": 0,
        "missing_or_invalid_query": 0,
        "missing_evidence": 0,
        "future_evidence": 0,
        "invalid_session_index": 0,
        "empty_history": 0,
    }
    for item in metadata:
        if not isinstance(item, dict):
            excluded["missing_or_invalid_query"] += 1
            continue
        uid = str(item.get("user_id", ""))
        user = users.get(uid)
        if user is None:
            excluded["unknown_user"] += 1
            continue
        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            excluded["missing_or_invalid_query"] += 1
            continue
        evidence = item.get("evidence_session_index")
        if not isinstance(evidence, list) or not evidence:
            excluded["missing_evidence"] += 1
            continue
        try:
            gold = {int(index) for index in evidence}
        except (TypeError, ValueError):
            excluded["invalid_session_index"] += 1
            continue

        sessions = user.get("sessions")
        timestamps = user.get("timestamps")
        session_types = user.get("session_types")
        if not (isinstance(sessions, list) and isinstance(timestamps, list)
                and isinstance(session_types, list)
                and len(sessions) == len(timestamps) == len(session_types)):
            raise ValueError(f"Inconsistent session arrays for user {uid}")

        query_time = _parse_time(item.get("query_timestamp"))
        turns = []
        visible_ids = set()
        for index, (session, timestamp, session_type) in enumerate(
            zip(sessions, timestamps, session_types)
        ):
            if index < 0:
                continue
            session_time = _parse_time(timestamp)
            if session_time > query_time:
                continue
            text = _session_text(session)
            if not text:
                continue
            visible_ids.add(index)
            turns.append({
                "turn_id": f"{uid}:{index:04d}",
                "text": f"SESSION_TYPE: {session_type}\nDATE: {timestamp}\n{text}",
                "relevant": index in gold,
            })

        if not turns:
            excluded["empty_history"] += 1
            continue
        if not gold.issubset(visible_ids):
            excluded["future_evidence"] += 1
            continue
        qtype = str(item.get("query_type", "unknown"))
        cases.append({
            "id": f"{uid}:{qtype}",
            "question": query.strip(),
            "question_type": QUERY_TYPES.get(qtype, qtype),
            "gold_turn_count": len(gold),
            "turns": turns,
        })

    summary = {
        "dataset": "PRAGMA",
        "users_in_history_file": len(users),
        "metadata_records": len(metadata),
        "eligible_questions": len(cases),
        "excluded_by_reason": excluded,
        "retrieval_unit": "whole timestamped session",
        "temporal_policy": "sessions strictly after query_timestamp are excluded; all gold evidence sessions must remain visible",
        "question_types": QUERY_TYPES,
    }
    return cases, summary


def paired_user_bootstrap(baseline, ablation, seed=20261127, resamples=20000):
    """Paired bootstrap by user, retaining all that user's questions per draw."""
    import random

    base = {row["question_id"]: row for row in baseline["per_question"]}
    abla = {row["question_id"]: row for row in ablation["per_question"]}
    ids = sorted(set(base) & set(abla))
    by_user = {}
    for qid in ids:
        uid = qid.rsplit(":", 1)[0]
        by_user.setdefault(uid, []).append(qid)
    if not ids:
        return {"matched_questions": 0, "user_clusters": 0, "metrics": {}}

    metrics = ("recall_at_5", "hit_at_5", "mrr", "recall_at_10", "hit_at_10")
    users = sorted(by_user)
    user_counts = {uid: len(by_user[uid]) for uid in users}
    user_sums = {
        uid: {
            metric: sum(abla[qid][metric] - base[qid][metric] for qid in by_user[uid])
            for metric in metrics
        }
        for uid in users
    }
    rng = random.Random(seed)
    draws = {metric: [] for metric in metrics}
    for _ in range(resamples):
        chosen = [rng.choice(users) for _ in users]
        denominator = sum(user_counts[uid] for uid in chosen)
        for metric in metrics:
            total = sum(user_sums[uid][metric] for uid in chosen)
            draws[metric].append(total / denominator)

    output = {
        "matched_questions": len(ids),
        "user_clusters": len(users),
        "bootstrap_unit": "user; sample users with replacement and retain all their questions",
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "metrics": {},
    }
    for metric in metrics:
        deltas = [abla[qid][metric] - base[qid][metric] for qid in ids]
        values = sorted(draws[metric])
        output["metrics"][metric] = {
            "mean_change": sum(deltas) / len(deltas),
            "improved": sum(delta > 1e-12 for delta in deltas),
            "regressed": sum(delta < -1e-12 for delta in deltas),
            "unchanged": sum(abs(delta) <= 1e-12 for delta in deltas),
            "user_cluster_bootstrap_95_ci": [
                values[int(0.025 * resamples)],
                values[min(resamples - 1, int(0.975 * resamples))],
            ],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--histories", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    parser.add_argument("--dataset-revision", default=None)
    parser.add_argument("--license-sha256", default=None)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")

    histories = json.loads(Path(args.histories).read_text(encoding="utf-8"))
    metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
    cases, build = build_pragma_cases(histories, metadata)
    if not cases:
        raise SystemExit(f"No eligible PRAGMA cases: {build}")

    ks = (1, 3, 5, 10)
    baseline = evaluate_turn_retrieval(cases, rank_bm25, ks)
    rrf = evaluate_turn_retrieval(cases, rank_rrf, ks)
    ablation = evaluate_turn_retrieval(
        cases,
        lambda query, sessions: rank_bm25(
            query, sessions, stopwords=BM25_ABLATION_STOPWORDS
        ),
        ks,
    )
    selective = evaluate_turn_retrieval(
        cases,
        lambda query, sessions: rank_bm25(
            query, sessions, stopwords=BM25_SELECTIVE_STOPWORDS
        ),
        ks,
    )
    jaccard = evaluate_turn_retrieval(cases, rank_jaccard, ks)
    paired = paired_user_bootstrap(
        baseline, ablation, resamples=args.bootstrap_resamples
    )
    paired_rrf = paired_user_bootstrap(
        baseline, rrf, resamples=args.bootstrap_resamples
    )
    paired_selective = paired_user_bootstrap(
        baseline, selective, resamples=args.bootstrap_resamples
    )
    summary = {
        "build": build,
        "source": {
            "dataset_repo": "stellahj/PRAGMA",
            "dataset_revision": args.dataset_revision,
            "license_sha256": args.license_sha256,
            "license": "NVIDIA License; non-commercial research/evaluation only",
        },
        "protocol": {
            "control": "Okapi BM25 k1=1.5, b=0.75, unchanged tokenizer",
            "ablation": "same BM25 parameters with the frozen LongMemEval stopword list",
            "selective_ablation": {
                "removed_terms": sorted(BM25_SELECTIVE_STOPWORDS),
                "preserved_cue_families": "temporal, relational, interrogative, possessive, and action terms",
                "status": "exploratory; not held-out validation",
            },
            "comparator": "token-set Jaccard",
            "cutoffs": list(ks),
            "retrieval_unit": "session",
            "time_filter": "only sessions at or before query_timestamp are searchable",
            "scope_warning": "Evidence-session retrieval only; no personalized-guidance response is generated or judged.",
        },
        "retrieval_jaccard": jaccard["metrics"],
        "retrieval_bm25": baseline["metrics"],
        "retrieval_rrf_bm25_jaccard": rrf["metrics"],
        "rrf_protocol": {"fusion": "reciprocal rank fusion of BM25 and token-set Jaccard", "k": 60},
        "retrieval_bm25_stopword_ablation": ablation["metrics"],
        "retrieval_bm25_selective_stopword_ablation": selective["metrics"],
        "paired_user_analysis": paired,
        "paired_rrf_user_analysis": paired_rrf,
        "paired_selective_user_analysis": paired_selective,
        "per_question": {
            "bm25": baseline["per_question"],
            "rrf_bm25_jaccard": rrf["per_question"],
            "stopword_ablation": ablation["per_question"],
            "selective_stopword_ablation": selective["per_question"],
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "build": build,
        "bm25": baseline["metrics"],
        "rrf_bm25_jaccard": rrf["metrics"],
        "stopword_ablation": ablation["metrics"],
        "selective_stopword_ablation": selective["metrics"],
        "paired": paired,
        "paired_selective": paired_selective,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
