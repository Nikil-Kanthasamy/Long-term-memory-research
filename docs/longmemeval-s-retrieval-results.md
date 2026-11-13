# LongMemEval S retrieval baseline results

## Run record

- Benchmark: LongMemEval S, cleaned split
- Dataset source: `xiaowu0162/longmemeval-cleaned`, file `longmemeval_s_cleaned.json`
- Questions evaluated: 500
- Retrieval questions with gold evidence: 470
- Abstention questions excluded from retrieval averages: 30
- Retrieval depth: K = 5
- Retrieval method: repository lexical Jaccard scorer with importance weighting
- Retrieval granularity: session
- Benchmark code revision: `f8a6e1a36944af8d24fa2d023a296273abbdaa62`
- GitHub Actions run: [43500251156](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43500251156)
- Run result: successful
- Machine-readable output: workflow artifact `longmemeval-s-retrieval-43500251156` (30-day retention)

The benchmark dataset and generated report are not committed to the repository. The workflow artifact contains the complete per-question output.

## Overall retrieval results

| Metric | Result |
|---|---:|
| Precision@5 | 8.51% |
| Recall@5 | 28.73% |
| Mean Reciprocal Rank (MRR) | 0.1991 |
| Hit@1 | 11.91% |

The 30 questions without gold evidence were excluded from retrieval metric averages and counted separately. Therefore, aggregate retrieval metrics use 470 questions, not all 500.

## Results by question type

| Question type | Scored questions | Precision@5 | Recall@5 | MRR | Hit@1 |
|---|---:|---:|---:|---:|---:|
| Single-session user | 64 | 5.94% | 29.69% | 0.1612 | 7.81% |
| Multi-session | 121 | 7.27% | 16.83% | 0.1364 | 7.44% |
| Single-session preference | 30 | 1.33% | 6.67% | 0.0178 | 0.00% |
| Temporal reasoning | 127 | 6.77% | 17.44% | 0.1154 | 3.94% |
| Knowledge update | 72 | 11.39% | 28.47% | 0.1637 | 5.56% |
| Single-session assistant | 56 | 18.21% | 91.07% | 0.7104 | 58.93% |

Question-type counts above are scored cases; abstention cases are not included in these denominators.

## Interpretation

1. The lexical baseline is weak overall: it retrieves at least one gold session in only 11.91% of scored questions at rank one.
2. Preference retrieval is a clear failure area in this baseline (MRR 0.0178; Hit@1 0%). This motivates testing preference-aware memory extraction and query expansion.
3. Multi-session and temporal reasoning also remain difficult, with Recall@5 below 18%.
4. Single-session assistant questions are a strong outlier. Their high scores should be investigated for dataset-specific lexical overlap or answer-session structure before drawing broad conclusions.
5. Knowledge-update retrieval has moderate recall but low Hit@1, suggesting relevant sessions are sometimes retrieved but not ranked first.

## Scope and limitations

- These are session-retrieval results only, not end-to-end answer accuracy.
- The system uses lexical Jaccard retrieval with importance weighting. It is not BM25, dense retrieval, or a reproduction of every LongMemEval baseline.
- No Gemini answer generation or official GPT-4o judging was performed in this run.
- Results are one baseline run at K=5. No ablation, confidence interval, significance test, or comparison to a paper baseline has been performed.
- Do not claim this result establishes superiority over other memory systems.

## Recommended next experiments

1. Add a BM25 baseline and compare it on the same full S split.
2. Add a dense embedding retrieval baseline, then a hybrid lexical+dense variant.
3. Prioritize preference, multi-session, and temporal-reasoning slices for error analysis.
4. Evaluate answer generation separately with a fixed reader model and the official LongMemEval judging script.
5. Preserve the dataset version, code SHA, retrieval settings, and artifact for every run.
