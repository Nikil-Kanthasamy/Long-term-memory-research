"""Offline retrieval evaluation for annotated query/memory relevance sets."""
import json
from pathlib import Path

def score_ranking(ranked_ids, relevant_ids, k=5):
    relevant = set(relevant_ids)
    ranked = list(ranked_ids[:k])
    hits = [1 if item in relevant else 0 for item in ranked]
    hit_count = sum(hits)
    precision = hit_count / max(1, k)
    recall = hit_count / len(relevant) if relevant else 1.0
    reciprocal_rank = next((1.0 / (i + 1) for i, hit in enumerate(hits) if hit), 0.0)
    return {"precision_at_k": precision, "recall_at_k": recall,
            "reciprocal_rank": reciprocal_rank, "hits": hit_count}

def evaluate_cases(store, cases, user_id, mode="hybrid", k=5, embed_fn=None):
    if mode not in {"lexical", "semantic", "hybrid"}:
        raise ValueError("mode must be lexical, semantic, or hybrid")
    rows = []
    for case in cases:
        query = str(case["query"])
        case_user = str(case.get("user_id", user_id))
        vector = None
        if mode in {"semantic", "hybrid"} and embed_fn:
            try:
                vector = embed_fn(query)
            except Exception:
                vector = None
        results = store.search(case_user, query, query_embedding=vector, limit=k, mode=mode)
        scores = score_ranking([int(item["id"]) for item in results],
                               [int(x) for x in case.get("relevant_memory_ids", [])], k)
        rows.append({"query": query, "user_id": case_user, **scores,
                     "retrieved_ids": [int(item["id"]) for item in results]})
    n = len(rows)
    metrics = {key: (sum(row[key] for row in rows) / n if n else 0.0)
               for key in ("precision_at_k", "recall_at_k", "reciprocal_rank")}
    return {"mode": mode, "k": k, "cases": n, "metrics": metrics, "results": rows}

def load_cases(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Evaluation JSON must contain a list of cases")
    return payload
