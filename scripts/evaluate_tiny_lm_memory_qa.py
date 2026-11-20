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
from src.ltm.tiny_lm import TinyLanguageModel, build_memory_prompt, tokenizer_from_checkpoint


def normalize(text):
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def token_f1(prediction, reference):
    """Compute token-set F1 after normalization; diagnostic, not semantic judging."""
    predicted = set(normalize(prediction).split())
    expected = set(normalize(reference).split())
    if not predicted or not expected:
        return 0.0
    overlap = len(predicted & expected)
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if overlap else 0.0


def retrieval_only_evidence(retrieved):
    """Return retrieved memory text without asking the language model to rewrite it."""
    return "\n".join(item.get("content", "") for item in retrieved if item.get("content"))


def contains_normalized_answer(text, expected_answer):
    """Match a complete contiguous token sequence, not an arbitrary substring."""
    observed_tokens = normalize(text).split()
    expected_tokens = normalize(expected_answer).split()
    if not expected_tokens or len(expected_tokens) > len(observed_tokens):
        return False
    width = len(expected_tokens)
    return any(
        observed_tokens[index:index + width] == expected_tokens
        for index in range(len(observed_tokens) - width + 1)
    )


def cluster_bootstrap_delta(results, condition, baseline, metric, *, replicates=2000, seed=2026):
    """Paired cluster bootstrap for condition-minus-baseline metric differences.

    Each fact cluster contributes one mean per condition. Bootstrap resampling
    happens over clusters, not paraphrase rows, so paraphrases are not treated
    as independent facts.
    """
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 1:
        raise ValueError("replicates must be a positive integer")
    grouped = {}
    for row in results:
        cluster_id = row.get("fact_cluster_id") or row["case_id"]
        if condition not in row or baseline not in row:
            continue
        grouped.setdefault(cluster_id, {"condition": [], "baseline": []})
        grouped[cluster_id]["condition"].append(row[condition][metric])
        grouped[cluster_id]["baseline"].append(row[baseline][metric])
    deltas = []
    for cluster_id in sorted(grouped):
        values = grouped[cluster_id]
        left = sum(values["condition"]) / len(values["condition"])
        right = sum(values["baseline"]) / len(values["baseline"])
        deltas.append(left - right)
    if not deltas:
        return {"clusters": 0, "mean_delta": None, "ci_95": None,
                "positive_delta_clusters": 0, "negative_delta_clusters": 0, "zero_delta_clusters": 0}
    observed = sum(deltas) / len(deltas)
    rng = __import__("random").Random(seed)
    boot_means = []
    for _ in range(replicates):
        sample = [deltas[rng.randrange(len(deltas))] for _ in deltas]
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    lower = boot_means[int(0.025 * (replicates - 1))]
    upper = boot_means[int(0.975 * (replicates - 1))]
    return {
        "clusters": len(deltas),
        "mean_delta": observed,
        "ci_95": [lower, upper],
        "positive_delta_clusters": sum(value > 0 for value in deltas),
        "negative_delta_clusters": sum(value < 0 for value in deltas),
        "zero_delta_clusters": sum(value == 0 for value in deltas),
        "metric_definition": f"mean({condition}.{metric}) - mean({baseline}.{metric}) within each fact cluster, then averaged across clusters",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
    }


def retrieval_contains_expected(retrieved, expected_answer):
    """Check whether retrieved evidence contains the complete normalized answer."""
    return any(
        contains_normalized_answer(item.get("content", ""), expected_answer)
        for item in retrieved
    )


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
        distractors = item.get("irrelevant_memories", [])
        if not isinstance(distractors, list) or not all(isinstance(x, str) and x.strip() for x in distractors):
            raise ValueError(f"line {number}: irrelevant_memories must be a list of non-empty strings")
        cases.append(item)
    if not cases:
        raise ValueError("no evaluation cases found")
    return cases


def build_condition_prompts(question, retrieved, block_size, tokenizer=None, distractor_memories=None):
    """Build no-memory, relevant-memory, and irrelevant-memory control prompts."""
    memories = [item.get("content", "") for item in retrieved if item.get("content")]
    prompts = {
        "no_memory": build_memory_prompt(
            question, [], block_size, include_memory=False, tokenizer=tokenizer
        ),
        "memory": build_memory_prompt(
            question, memories, block_size, include_memory=True, tokenizer=tokenizer
        ),
    }
    if distractor_memories:
        prompts["irrelevant_memory"] = build_memory_prompt(
            question, distractor_memories, block_size, include_memory=True, tokenizer=tokenizer
        )
    return prompts


def generate_answer(model, tokenizer, prompt, max_new_tokens, device):
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if ids.size(1) > model.config.block_size:
        ids = ids[:, -model.config.block_size:]
    # Record the actual prompt boundary after any context-window truncation.
    prompt_length = ids.size(1)
    with torch.no_grad():
        model.eval()
        for _ in range(max_new_tokens):
            context = ids[:, -model.config.block_size:]
            logits, _ = model(context)
            next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            ids = torch.cat((ids, next_id), dim=1)
            if tokenizer.decode(ids[0, prompt_length:].tolist()).endswith("\n"):
                break
    return tokenizer.decode(ids[0, prompt_length:].tolist()).strip()



def score_expected_response(model, tokenizer, prompt, response, device):
    """Score gold response tokens by teacher-forced mean NLL after a prompt.

    This is a diagnostic of conditional probability, not free-form answer
    accuracy. The prompt is left-truncated only if needed to keep the complete
    gold response inside the model context window.
    """
    prompt_ids = tokenizer.encode(prompt)
    response_ids = tokenizer.encode(response.rstrip() + "\n")
    block_size = model.config.block_size
    if block_size < 2:
        raise ValueError("model block_size must be at least 2")
    if len(response_ids) >= block_size:
        response_ids = response_ids[-(block_size - 1):]
        prompt_ids = prompt_ids[-1:]
    else:
        prompt_ids = prompt_ids[-(block_size - len(response_ids)):]
    combined = prompt_ids + response_ids
    if not prompt_ids or not response_ids:
        raise ValueError("prompt and response must each contain at least one token")
    input_ids = torch.tensor([combined[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor(combined[1:], dtype=torch.long, device=device)
    first_response_logit = len(prompt_ids) - 1
    with torch.no_grad():
        model.eval()
        logits, _ = model(input_ids)
        response_logits = logits[:, first_response_logit:, :]
        response_targets = targets[first_response_logit:]
        nll = torch.nn.functional.cross_entropy(
            response_logits.reshape(-1, model.config.vocab_size),
            response_targets,
            reduction="mean",
        )
    return {
        "mean_nll": float(nll.item()),
        "perplexity": float(torch.exp(nll).item()),
        "token_count": len(response_ids),
    }

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
    tokenizer = tokenizer_from_checkpoint(checkpoint)
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
        distractor_memories = case.get("irrelevant_memories", [])
        distractor_source = "case_curated" if distractor_memories else "cross_case_fallback"
        if not distractor_memories:
            for offset in range(1, len(cases)):
                candidate = cases[(index + offset) % len(cases)].get("memories", [])
                if candidate and candidate != case.get("memories", []):
                    distractor_memories = [candidate[0]]
                    break
        prompts = build_condition_prompts(
            case["question"], retrieved, model.config.block_size, tokenizer,
            distractor_memories=distractor_memories,
        )
        condition_results = {}
        for condition, prompt in prompts.items():
            torch.manual_seed(1000 + index)
            expected_score = score_expected_response(
                model, tokenizer, prompt, case["expected_answer"], device
            )
            answer = generate_answer(model, tokenizer, prompt, args.max_new_tokens, device)
            matched = contains_normalized_answer(answer, case["expected_answer"])
            condition_results[condition] = {
                "answer": answer,
                "matched": matched,
                "token_f1": token_f1(answer, case["expected_answer"]),
                "gold_answer_score": expected_score,
            }
        results.append({
            "case_id": case.get("id", str(index)),
            "fact_cluster_id": case.get("fact_cluster_id", case.get("id", str(index))),
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "retrieved_memory_count": len(retrieved),
            "retrieval_hit": retrieval_contains_expected(retrieved, case["expected_answer"]),
            "irrelevant_memory_source": distractor_source if distractor_memories else None,
            "retrieval_only_evidence": {
                "evidence": retrieval_only_evidence(retrieved),
                "matched": retrieval_contains_expected(retrieved, case["expected_answer"]),
            },
            **condition_results,
        })

    summary = {}
    conditions = ("no_memory", "memory") + (("irrelevant_memory",) if any("irrelevant_memory" in row for row in results) else ())
    for condition in conditions:
        hits = sum(row[condition]["matched"] for row in results if condition in row)
        total = sum(condition in row for row in results)
        summary[condition] = {
            "correct": hits,
            "total": total,
            "exact_contains_accuracy": hits / total if total else 0.0,
            "mean_token_f1": (
                sum(row[condition]["token_f1"] for row in results if condition in row) / total
                if total else 0.0
            ),
        }
    retrieval_hits = sum(row["retrieval_hit"] for row in results)
    evidence_hits = sum(row["retrieval_only_evidence"]["matched"] for row in results)
    gold_answer_scores = {}
    for condition in conditions:
        scores = [row[condition]["gold_answer_score"] for row in results if condition in row]
        gold_answer_scores[condition] = {
            "mean_nll": sum(item["mean_nll"] for item in scores) / len(scores),
            "mean_perplexity": sum(item["perplexity"] for item in scores) / len(scores),
            "total_scored_tokens": sum(item["token_count"] for item in scores),
        }
    gold_answer_nll_delta_memory_minus_no_memory = (
        gold_answer_scores["memory"]["mean_nll"]
        - gold_answer_scores["no_memory"]["mean_nll"]
    )
    irrelevant_memory_nll_delta = (
        gold_answer_scores["irrelevant_memory"]["mean_nll"]
        - gold_answer_scores["no_memory"]["mean_nll"]
        if "irrelevant_memory" in gold_answer_scores else None
    )
    cluster_analysis = {
        "independent_fact_clusters": len({row["fact_cluster_id"] for row in results}),
        "memory_minus_no_memory": {
            "exact_match_accuracy": cluster_bootstrap_delta(
                results, "memory", "no_memory", "matched"
            ),
            "token_f1": cluster_bootstrap_delta(
                results, "memory", "no_memory", "token_f1"
            ),
        },
    }
    # Aggregate teacher-forced NLL per cluster before paired resampling.
    nll_cluster_rows = []
    nll_grouped = {}
    for row in results:
        cluster_id = row["fact_cluster_id"]
        nll_grouped.setdefault(cluster_id, {"memory": [], "no_memory": []})
        for condition in ("memory", "no_memory"):
            nll_grouped[cluster_id][condition].append(
                row[condition]["gold_answer_score"]["mean_nll"]
            )
    for cluster_id, values in nll_grouped.items():
        nll_cluster_rows.append({
            "case_id": cluster_id,
            "fact_cluster_id": cluster_id,
            "memory": {"nll": sum(values["memory"]) / len(values["memory"])},
            "no_memory": {"nll": sum(values["no_memory"]) / len(values["no_memory"])},
        })
    cluster_analysis["memory_minus_no_memory"]["gold_answer_mean_nll"] = (
        cluster_bootstrap_delta(nll_cluster_rows, "memory", "no_memory", "nll")
    )
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
        "retrieval_only_evidence": {
            "correct": evidence_hits,
            "total": len(results),
            "evidence_contains_gold_accuracy": evidence_hits / len(results),
        },
        "cluster_analysis": cluster_analysis,
        "gold_answer_likelihood": {
            "by_condition": gold_answer_scores,
            "memory_minus_no_memory_mean_nll": gold_answer_nll_delta_memory_minus_no_memory,
            "irrelevant_memory_minus_no_memory_mean_nll": irrelevant_memory_nll_delta,
            "interpretation": "Compare relevant and irrelevant memory deltas against no memory. Teacher-forced NLL is not generated-answer accuracy.",
        },
        "results": results,
        "limitations": [
            "Synthetic evaluation facts are manually authored; results do not establish performance on real user memories.",
            "Fact-cluster bootstrap resamples independent fact clusters, not paraphrase rows; intervals remain exploratory for this small synthetic benchmark.",
            "Generation exact-match scoring requires a complete contiguous sequence of normalized answer tokens; paraphrases may still be marked wrong. Mean token-set F1 is also reported as a partial-overlap diagnostic, not semantic judging.",
            "Retrieval hit checks literal gold-answer presence in retrieved text, not semantic evidence quality.",
            "A model that generates gibberish may score zero even when retrieval succeeds.",
            "retrieval_only_evidence is a diagnostic evidence-exposure baseline, not a generated natural-language answer.",
            "gold_answer_likelihood is teacher-forced token likelihood, not free-form answer accuracy or proof of semantic reasoning.",
            "Curated irrelevant_memories are manually selected distractors intended to be topic-matched but answer-irrelevant; cross-case fallback distractors are not guaranteed semantically unrelated.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "retrieval_summary": report["retrieval_summary"], "cluster_analysis": report["cluster_analysis"], "gold_answer_likelihood": report["gold_answer_likelihood"], "output": str(output)}, indent=2))
    if temporary:
        temporary.cleanup()


if __name__ == "__main__":
    main()
