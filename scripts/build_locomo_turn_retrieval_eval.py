"""Evaluate frozen BM25 stopword ablation on the independent LoCoMo benchmark.

This retrieval-only evaluation uses the official QA evidence dialog IDs as
relevance labels. It does not evaluate answer generation or QA accuracy.
Dataset files are downloaded by CI and are never committed or uploaded.
"""
import argparse
import json
import re
from collections import Counter
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
    # Direct script execution places scripts/ (not the repository root) on sys.path.
    from build_longmemeval_turn_retrieval_eval import (
        BM25_ABLATION_STOPWORDS,
        BM25_SELECTIVE_STOPWORDS,
        evaluate_turn_retrieval,
        rank_bm25,
        rank_jaccard,
    )


CATEGORY_NAMES = {
    1: "single-hop",
    2: "temporal",
    3: "multi-hop",
    4: "open-domain",
    5: "adversarial",
}


def _session_number(key):
    match = re.fullmatch(r"session_(\d+)", str(key))
    return int(match.group(1)) if match else None


def build_locomo_cases(dataset):
    """Normalize LoCoMo conversation/session and QA evidence annotations."""
    cases = []
    counts = Counter()
    if not isinstance(dataset, list):
        raise ValueError("LoCoMo input must be a JSON array")

    for sample in dataset:
        sample_id = str(sample.get("sample_id", "unknown"))
        conversation = sample.get("conversation")
        if not isinstance(conversation, dict):
            counts["invalid_conversation"] += len(sample.get("qa") or [])
            continue

        sessions = sorted(
            ((number, key, value) for key, value in conversation.items()
             if (number := _session_number(key)) is not None and isinstance(value, list)),
            key=lambda item: item[0],
        )
        turns = []
        for number, key, session_turns in sessions:
            date = str(conversation.get(f"{key}_date_time", ""))
            for turn in session_turns:
                if not isinstance(turn, dict):
                    continue
                text = turn.get("text")
                dia_id = turn.get("dia_id")
                if not isinstance(text, str) or not text.strip() or dia_id is None:
                    continue
                turns.append({
                    "turn_id": str(dia_id),
                    "session_id": key,
                    "session_index": number - 1,
                    "turn_index": len(turns),
                    "role": str(turn.get("speaker", "unknown")),
                    "date": date,
                    "text": text.strip(),
                    "relevant": False,
                })
        turn_ids = {turn["turn_id"] for turn in turns}
        if not turns:
            counts["conversation_without_turns"] += len(sample.get("qa") or [])
            continue

        for qa_index, qa in enumerate(sample.get("qa") or []):
            if not isinstance(qa, dict):
                counts["invalid_qa"] += 1
                continue
            category = qa.get("category")
            try:
                category_num = int(category)
            except (TypeError, ValueError):
                category_num = 0
            if category_num == 5 or not qa.get("answer"):
                counts["abstention_or_adversarial"] += 1
                continue
            question = qa.get("question")
            if not isinstance(question, str) or not question.strip():
                counts["invalid_question"] += 1
                continue
            evidence = qa.get("evidence")
            if not isinstance(evidence, list) or not evidence:
                counts["missing_evidence"] += 1
                continue
            evidence_ids = {str(value) for value in evidence}
            found_evidence = evidence_ids & turn_ids
            if not found_evidence:
                counts["evidence_id_not_found"] += 1
                continue

            case_turns = [
                {**turn, "relevant": turn["turn_id"] in evidence_ids}
                for turn in turns
            ]
            cases.append({
                "id": f"{sample_id}:{qa_index}",
                "question": question.strip(),
                "question_type": CATEGORY_NAMES.get(category_num, f"category-{category_num}"),
                "gold_turn_count": len(found_evidence),
                "turns": case_turns,
            })
            counts["eligible_questions"] += 1

    summary = {
        "dataset": "LoCoMo",
        "conversations": len(dataset),
        "eligible_questions": len(cases),
        "excluded_by_reason": dict(sorted(
            (key, value) for key, value in counts.items() if key != "eligible_questions"
        )),
        "turns_per_question_min": min((len(case["turns"]) for case in cases), default=0),
        "turns_per_question_max": max((len(case["turns"]) for case in cases), default=0),
        "relevance": "LoCoMo QA evidence dialog IDs; retrieval-only, no answer judging",
    }
    return cases, summary


def paired_summary(baseline, ablation, seed=20261127, resamples=20000):
    """Paired outcomes with conversation-cluster bootstrap confidence intervals."""
    import random

    baseline_rows = {row["question_id"]: row for row in baseline["per_question"]}
    ablation_rows = {row["question_id"]: row for row in ablation["per_question"]}
    ids = sorted(set(baseline_rows) & set(ablation_rows))
    if not ids:
        return {"matched_questions": 0, "metrics": {}}

    metrics = ("recall_at_5", "hit_at_5", "mrr", "recall_at_10", "hit_at_10")
    by_conversation = {}
    for qid in ids:
        conversation_id = qid.rsplit(":", 1)[0]
        bucket = by_conversation.setdefault(
            conversation_id, {metric: [] for metric in metrics}
        )
        for metric in metrics:
            bucket[metric].append(
                ablation_rows[qid][metric] - baseline_rows[qid][metric]
            )

    cluster_ids = sorted(by_conversation)
    rng = random.Random(seed)
    samples = {metric: [] for metric in metrics}
    for _ in range(resamples):
        selected = [rng.choice(cluster_ids) for _ in cluster_ids]
        denominators = sum(len(by_conversation[c][metrics[0]]) for c in selected)
        for metric in metrics:
            total = sum(
                sum(by_conversation[c][metric]) for c in selected
            )
            samples[metric].append(total / denominators)

    output = {
        "matched_questions": len(ids),
        "conversation_clusters": len(cluster_ids),
        "bootstrap_unit": "conversation; sample conversations with replacement and retain all their questions",
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "metrics": {},
    }
    for metric in metrics:
        deltas = [
            ablation_rows[qid][metric] - baseline_rows[qid][metric]
            for qid in ids
        ]
        wins = sum(delta > 1e-12 for delta in deltas)
        losses = sum(delta < -1e-12 for delta in deltas)
        ties = len(deltas) - wins - losses
        values = sorted(samples[metric])
        output["metrics"][metric] = {
            "mean_change": sum(deltas) / len(deltas),
            "improved": wins,
            "regressed": losses,
            "unchanged": ties,
            "conversation_cluster_bootstrap_95_ci": [
                values[int(0.025 * resamples)],
                values[min(resamples - 1, int(0.975 * resamples))],
            ],
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_resamples < 100:
        parser.error("--bootstrap-resamples must be at least 100")
    dataset = json.loads(Path(args.input).read_text(encoding="utf-8"))
    cases, build = build_locomo_cases(dataset)
    if not cases:
        raise SystemExit("No eligible LoCoMo evidence-retrieval questions found")

    ks = (1, 3, 5, 10)
    baseline = evaluate_turn_retrieval(cases, rank_bm25, ks)
    rrf = evaluate_turn_retrieval(cases, rank_rrf, ks)
    ablation = evaluate_turn_retrieval(
        cases,
        lambda question, turns: rank_bm25(
            question, turns, stopwords=BM25_ABLATION_STOPWORDS
        ),
        ks,
    )
    selective = evaluate_turn_retrieval(
        cases,
        lambda question, turns: rank_bm25(
            question, turns, stopwords=BM25_SELECTIVE_STOPWORDS
        ),
        ks,
    )
    jaccard = evaluate_turn_retrieval(cases, rank_jaccard, ks)
    summary = {
        "build": build,
        "protocol": {
            "baseline": "Okapi BM25 k1=1.5, b=0.75, unchanged tokenizer",
            "ablation": "same BM25 parameters with the frozen LongMemEval stopword list",
            "selective_ablation": {
                "removed_terms": sorted(BM25_SELECTIVE_STOPWORDS),
                "preserved_cue_families": "temporal, relational, interrogative, possessive, and action terms",
                "status": "exploratory; not held-out validation",
            },
            "top_k": list(ks),
            "unit": "individual dialogue turn",
            "relevance": "turn dia_id listed in the QA evidence field",
            "scope_warning": "Retrieval of annotated evidence only; not QA accuracy or answer generation.",
        },
        "retrieval_jaccard": jaccard["metrics"],
        "retrieval_bm25": baseline["metrics"],
        "retrieval_rrf_bm25_jaccard": rrf["metrics"],
        "rrf_protocol": {"fusion": "reciprocal rank fusion of BM25 and token-set Jaccard", "k": 60},
        "retrieval_bm25_stopword_ablation": ablation["metrics"],
        "retrieval_bm25_selective_stopword_ablation": selective["metrics"],
        "paired_question_analysis": paired_summary(
            baseline, ablation, resamples=args.bootstrap_resamples
        ),
        "paired_rrf_question_analysis": paired_summary(
            baseline, rrf, resamples=args.bootstrap_resamples
        ),
        "paired_selective_question_analysis": paired_summary(
            baseline, selective, resamples=args.bootstrap_resamples
        ),
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
        "paired": summary["paired_question_analysis"],
        "paired_selective": summary["paired_selective_question_analysis"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
