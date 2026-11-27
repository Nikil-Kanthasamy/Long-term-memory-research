# LongMemEval-S turn-level evidence retrieval results

- Workflow: [run 45925646437](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45925646437)
- Evaluated revision: `dbe74c454a6b0878facd555bc824643b642dd98e`
- Dataset: official cleaned LongMemEval-S, 500 questions
- Retrieval-only; no answer generation or semantic judging

## Coverage and protocol

The benchmark built an index of **231,595 non-empty conversation turns** and evaluated all **470 answerable questions**. It retained **886 labeled evidence turns**. All 470 answerable questions were eligible (100% coverage); none were skipped. The remaining 30 questions are benchmark abstentions and are excluded from retrieval averages.

A turn is relevant only when its official LongMemEval turn annotation has `has_answer=true`. Both methods rank individual turns within each question's history. Jaccard is token-set overlap; BM25 is Okapi BM25 (k1=1.5, b=0.75). These are evidence-retrieval metrics, not answer accuracy.

## Aggregate turn-level results

| Metric | BM25 | Jaccard |
|---|---:|---:|
| Precision@1 | 55.32% | 28.72% |
| Recall@1 | 37.13% | 17.22% |
| Hit@1 | 55.32% | 28.72% |
| Precision@5 | 22.77% | 14.38% |
| Recall@5 | 67.15% | 41.72% |
| Hit@5 | 79.36% | 52.55% |
| Precision@10 | 13.21% | 9.06% |
| Recall@10 | 76.23% | 51.76% |
| Hit@10 | 84.89% | 62.34% |
| MRR | 66.30% | 40.01% |

At K=5, BM25 retrieves at least one labeled evidence turn for 79.36% of answerable questions, and retrieves 67.15% of all labeled evidence turns on average. The difference between Hit@5 and Recall@5 matters: a question can have multiple gold evidence turns, so finding one does not mean finding all of them.

## BM25 results by question type

| Question type | Questions | Recall@5 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| knowledge-update | 72 | 81.48% | 90.28% | 74.48% |
| multi-session | 121 | 47.36% | 76.03% | 59.00% |
| single-session-assistant | 56 | 85.71% | 85.71% | 78.90% |
| single-session-preference | 30 | 33.33% | 40.00% | 22.07% |
| single-session-user | 64 | 89.84% | 90.62% | 80.97% |
| temporal-reasoning | 127 | 66.25% | 77.17% | 66.12% |

Multi-session and preference questions are the clearest retrieval weaknesses in this split. The per-type results are descriptive; no confidence intervals or significance tests were computed for this run.

## Session-level comparison from the same workflow

The same run also reproduced session-level retrieval over the same 470 answerable questions:

| Metric | Session BM25 | Session lexical |
|---|---:|---:|
| Precision@5 | 33.32% | 9.45% |
| Recall@5 | 91.44% | 32.09% |
| MRR | 90.75% | 20.98% |
| Hit@1 | 87.02% | 11.70% |

Do not compare session Recall@5 directly with turn Recall@5 as though they were the same task: the units and relevance labels differ. Session retrieval asks whether the correct conversation session is found; turn retrieval asks whether the specific annotated evidence turns are surfaced. The turn-level result exposes a remaining evidence-localization bottleneck even when session-level retrieval is strong.

## Limitations and next experiments

- The relevance target is the benchmark's exact `has_answer` annotation, not a semantic relevance judgment. An unannotated but useful neighboring turn is treated as non-relevant.
- Retrieval scores do not show that a reader can synthesize multiple turns, resolve temporal updates, or produce the correct answer.
- The current rankers use lexical terms only. Temporal metadata is retained in case output but this experiment does not establish robust temporal reasoning.
- Multi-session and preference categories deserve targeted error analysis, followed by controlled tests of query expansion, temporal features, and hybrid reranking.
- The next end-to-end milestone remains answer generation over retrieved evidence and official LongMemEval judging. Report its answer accuracy separately from these retrieval metrics.

The workflow uploaded the full per-question turn cases and summary as artifact `longmemeval-s-retrieval-comparison-45925646437`.


## Follow-up failure analysis — workflow run 45926510628

A follow-up run added rank-bucket counts and readable difficult examples. The 470-question turn-level BM25 metrics reproduced the earlier run (Recall@5 67.15%, Hit@5 79.36%, MRR 66.30%), while the new analysis makes the remaining ranking errors more concrete.

| Rank of first annotated evidence turn | Questions |
|---|---:|
| Rank 1 | 260 |
| Ranks 2–5 | 113 |
| Ranks 6–10 | 26 |
| Rank 11 or lower in the ranking (outside top 10) | 71 |
| No annotated evidence turn in the ranked corpus | 0 |

These buckets describe the position of the **first** annotated evidence turn, not the number of relevant turns retrieved. Since the evaluator ranks the full turn corpus, "not retrieved" is zero by construction when all gold turns are in the indexed corpus; the practically important miss bucket is 71 questions whose first gold turn is ranked below 10.

The 25 worst examples were selected by the lowest first-gold rank. Their question-type mix was 10 temporal-reasoning, 7 single-session-preference, 4 multi-session, 3 single-session-assistant, and 1 single-session-user. This is a diagnostic sample, not a representative estimate of category-wide failure rates. The category-level metrics above remain the better aggregate comparison.

Examples illustrate distinct query/evidence mismatches:
- A guitar-shopping advice question had its annotated evidence turn about upgrading from a Fender Stratocaster to a Gibson Les Paul at rank 478.
- A question asking for a business milestone mentioned four weeks earlier had its contract-signing evidence turn at rank 452.
- A gardening activity question ("two weeks ago") had its tomato-sapling evidence turn at rank 447.

The first example has a preference/advice intent whose evidence uses different wording; the latter two require linking a relative-time expression to an event description. These examples motivate separate controlled experiments: (1) query expansion or semantic/hybrid retrieval for paraphrased preference evidence, and (2) explicit date/temporal normalization for relative-time questions. They do **not** establish that either intervention will improve the benchmark; each should be compared against the unchanged BM25 baseline on the same 470 questions.

Artifact: `longmemeval-s-retrieval-comparison-45926510628` (workflow run [45926510628](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45926510628)).
