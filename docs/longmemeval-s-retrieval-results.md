# LongMemEval S retrieval baseline results

## Run record

- Benchmark: LongMemEval S, cleaned split
- Dataset source: `xiaowu0162/longmemeval-cleaned`, file `longmemeval_s_cleaned.json`
- Questions evaluated: 500
- Retrieval questions with gold evidence: 470
- Abstention questions excluded from retrieval averages: 30
- Retrieval depth: K = 5
- Retrieval granularity: session
- Paired benchmark run: [GitHub Actions run 43500776110](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43500776110)
- Run result: successful
- Code revision: `f1277d50362ea4fd59d9950adf3c38454c806542`
- Machine-readable artifact: `longmemeval-s-retrieval-comparison-43500776110` (30-day retention)

The paired workflow downloaded one dataset and ran both retrieval modes with the same questions and K. The dataset and generated reports are not committed to the repository; the artifact contains both per-question JSON reports.

## Overall retrieval results

| Metric | Lexical Jaccard | BM25 |
|---|---:|---:|
| Precision@5 | 8.51% | **33.32%** |
| Recall@5 | 28.73% | **91.44%** |
| Mean Reciprocal Rank (MRR) | 0.1991 | **0.9075** |
| Hit@1 | 11.91% | **87.02%** |

Both reports contain 500 questions: 470 with gold evidence and 30 abstention cases. Retrieval averages use the 470 evidence-bearing questions. BM25 scores each session using Okapi term-frequency, inverse-document-frequency, and document-length normalization; the existing importance bonus is applied only to sessions with a positive BM25 score. Importance is constant in this benchmark fixture, so it does not explain the ranking improvement.

## BM25 results by question type

| Question type | Scored questions | Precision@5 | Recall@5 | MRR | Hit@1 |
|---|---:|---:|---:|---:|---:|
| Single-session user | 64 | 20.00% | 100.00% | 0.9602 | 93.75% |
| Multi-session | 121 | 42.48% | 83.77% | 0.9132 | 88.43% |
| Single-session preference | 30 | 17.33% | 86.67% | 0.5661 | 40.00% |
| Temporal reasoning | 127 | 37.64% | 88.11% | 0.8806 | 83.46% |
| Knowledge update | 72 | 39.17% | 97.92% | 0.9688 | 94.44% |
| Single-session assistant | 56 | 20.00% | 100.00% | 1.0000 | 100.00% |

## Interpretation

1. BM25 is a large improvement over the repository's original lexical Jaccard baseline on this paired run. It retrieves at least one gold session at rank one for 87.02% of evidence-bearing questions.
2. Preference questions remain the weakest BM25 slice (Hit@1 40.00%, MRR 0.5661), so preference extraction and preference-aware query expansion still deserve focused work.
3. These are retrieval-only metrics. They do not show that a reader LLM can synthesize evidence, resolve contradictions, answer correctly, or abstain appropriately.
4. The result is a single deterministic run on LongMemEval-S at K=5. It has not been independently reproduced, tested on LongMemEval-M, or compared under identical conditions with published systems.
5. Before presenting this as a strong benchmark result, run per-question error analysis and audit the implementation against the official LongMemEval retrieval protocol. In particular, confirm duplicate-session handling, tie-breaking, ranking depth, and that no benchmark-only metadata enters indexed text.

## Scope and limitations

- Session-level retrieval only; no end-to-end QA accuracy has been measured.
- No Gemini answer generation or official LongMemEval GPT-4o judging was performed in this run.
- No dense embedding retrieval, hybrid reranking, ablation, confidence interval, or significance test has been performed.
- This is not a reproduction of a published LongMemEval system and does not establish superiority over other memory systems.
- The benchmark adapter indexes the session date and role-tagged message text. It does not index the `has_answer` annotation field.
- The workflow artifact is retained for 30 days; preserve it externally if a longer-lived experimental record is needed.

## Recommended next experiments

1. Audit BM25 results per question and verify the official scoring protocol.
2. Add dense embedding retrieval and compare lexical, BM25, dense, and hybrid retrieval on the same split.
3. Add a fixed reader LLM and evaluate generated answers with the official LongMemEval judge, including abstention cases.
4. Add error analysis for preference, temporal reasoning, and multi-session questions.
5. Preserve dataset version, code SHA, model identifiers, prompts, settings, costs, and artifacts for every run.

## Next milestone: end-to-end memory assistant

A complete LLM-powered long-term-memory assistant requires more than retrieval: session ingestion, fact extraction, memory update/supersession, hybrid retrieval, evidence-aware context assembly, an LLM reader, abstention behavior, and end-to-end evaluation. The current repository has several of these building blocks, but this benchmark only exercised retrieval. A first credible end-to-end LongMemEval-S result is estimated at 2–4 focused engineering days once a working LLM API key and quota are available; a more robust hybrid system with error analysis and repeatable evaluation is approximately 1–2 weeks. These are planning estimates, not measured completion times.
