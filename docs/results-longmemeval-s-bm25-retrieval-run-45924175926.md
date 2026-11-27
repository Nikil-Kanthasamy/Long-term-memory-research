# LongMemEval-S full-session retrieval: BM25 vs lexical baseline

**Run:** [GitHub Actions #45924175926](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45924175926)  
**Evaluated commit:** `f8b67b534970cd0aa79b42346172499e46db0785`  
**Dataset:** official cleaned LongMemEval-S, 500 questions  
**Retrieval granularity:** full conversation sessions  
**Cutoff:** top 5 sessions (`k=5`)  
**Scored:** 470 questions; 30 abstention questions were excluded from retrieval metrics because they have no gold evidence session.

## Overall results

| Metric | BM25 | Lexical baseline | BM25 gain |
|---|---:|---:|---:|
| Precision@5 | 33.32% | 9.45% | +23.87 pp |
| Recall@5 | **91.44%** | 32.09% | **+59.35 pp** |
| MRR | **90.75%** | 20.98% | **+69.77 pp** |
| Hit@1 | **87.02%** | 11.70% | **+75.32 pp** |

BM25 ranked a relevant evidence session first for 409 of the 470 scored questions (87.02%). It returned at least one relevant session in the top five for 455/470 questions (96.81%); recall@5, which accounts for questions with multiple gold sessions, was 91.44%.

## Paired uncertainty estimates

A paired nonparametric bootstrap was computed over the same 470 scored question IDs for both retrieval modes (20,000 resamples, fixed RNG seed 20261127). The intervals below are percentile 95% confidence intervals for the per-question BM25-minus-lexical metric difference.

| Metric difference | Mean gain | Paired bootstrap 95% CI |
|---|---:|---:|
| Precision@5 | +23.87 pp | [+22.09, +25.66] pp |
| Recall@5 | +59.35 pp | [+55.50, +63.16] pp |
| MRR | +69.77 pp | [+66.44, +73.01] pp |
| Hit@1 | +75.32 pp | [+71.28, +79.15] pp |

No bootstrap replicate had a non-positive difference for any of these metrics. These intervals quantify question-sampling uncertainty within this benchmark split; they do not establish generalization to other datasets or retrieval distributions.

## Recall@5 and MRR by question type

| Question type | Questions | BM25 Recall@5 | Lexical Recall@5 | BM25 MRR | Lexical MRR |
|---|---:|---:|---:|---:|---:|
| single-session-user | 64 | 100.00% | 40.63% | 96.02% | 17.97% |
| multi-session | 121 | 83.77% | 18.83% | 91.32% | 15.07% |
| single-session-preference | 30 | 86.67% | 6.67% | 56.61% | 2.33% |
| temporal-reasoning | 127 | 88.11% | 20.49% | 88.06% | 13.85% |
| knowledge-update | 72 | 97.92% | 29.17% | 96.88% | 16.50% |
| single-session-assistant | 56 | 100.00% | 94.64% | 100.00% | 69.11% |

The six question-type counts sum to all 470 scored questions.

## Interpretation

- BM25 is a substantially stronger lexical retrieval baseline than the existing overlap/Jaccard-style lexical mode on this benchmark.
- Gains appear across the major question types, including multi-session, temporal reasoning, preference, and knowledge-update questions.
- The single-session-assistant category is a relative exception: the existing lexical baseline already performs strongly there.
- This benchmark ranks complete haystack sessions using each question's own haystack. It does not measure retrieval across a persistent cross-user memory corpus.
- The evaluation measures whether gold evidence sessions are retrieved. It does not measure answer generation, factual correctness of generated answers, temporal synthesis quality, or whether a learned semantic retriever would outperform BM25.
- The 30 abstention questions are reported separately and are not included in the 470-question retrieval metric denominators.

## Reproduction

The workflow downloads the cleaned LongMemEval-S dataset, then runs the repository CLI twice with `--retrieval-mode lexical` and `--retrieval-mode bm25`, using `k=5`. The two JSON outputs are uploaded as the workflow artifact:

[Download workflow artifact](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45924175926)

The output files are `longmemeval_s_lexical.json` and `longmemeval_s_bm25.json`. The JSON files contain per-question ranked sessions and metrics, allowing independent recomputation.

## Next experiments

1. Add BM25 parameter and tokenizer ablations while retaining the same 470 scored questions.
2. Add hybrid BM25 + semantic retrieval, with identical question splits and explicit retrieval-cost reporting.
3. Evaluate answer generation from the top-k retrieved sessions separately from retrieval metrics.
4. Replicate the paired comparison on another LongMemEval split or a held-out conversational-memory dataset before making generalization claims.
