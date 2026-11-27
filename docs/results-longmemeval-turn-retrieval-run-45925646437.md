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
