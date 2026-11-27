# LongMemEval-S BM25 stopword ablation — reproducible run 45928603603

## Reproduction

- Workflow: [45928603603](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45928603603)
- Branch: `research/long-term-memory-starter`
- Commit: `a916aa71d8489848a2cd290282404e6bb39295bc`
- Workflow result: all steps passed, including unit-independent full dataset retrieval evaluation and artifact upload.
- Artifact: `longmemeval-s-retrieval-comparison-45928603603` (artifact ID 15167970011; 76,109,591 bytes; retained through 2026-12-27).
- Dataset: official cleaned LongMemEval-S; 500 questions, 470 answerable and 30 abstentions.
- Corpus: 231,595 nonempty turns and 886 labeled evidence turns.
- Coverage: 470/470 answerable questions; no skipped answerable questions.
- Relevance: official turn-level `has_answer=true` labels.
- Scope: retrieval of evidence turns only; not answer generation, answer correctness, or end-to-end assistant quality.

## Implementation

The benchmark script now reports both the unchanged default BM25 and an explicit `retrieval_bm25_stopword_ablation` condition. The ablation removes the following fixed list from both query and document terms:

`what, was, the, i, my, a, an, is, did, do, of, in, on, to, for, me, you, and, or, it, that, one, how, many, which, who, when, where, from, with, about, ago, have, has, had, mentioned, tell, can`

All other BM25 settings remain unchanged (Okapi k1=1.5, b=0.75). The default BM25 path is unchanged; an empty stopword set preserves its prior behavior. Unit tests cover this invariant and the explicit ablation list.

## Overall results

| Metric | Default BM25 | Stopword ablation | Paired change |
|---|---:|---:|---:|
| Recall@5 | 67.15% | 69.19% | +2.04 pp |
| Hit@5 | 79.36% | 81.28% | +1.91 pp |
| MRR | 66.30% | 66.17% | -0.13 pp |
| Recall@10 | 76.23% | 78.90% | +2.67 pp |
| Hit@10 | 84.89% | 87.23% | +2.34 pp |

The results reproduce the earlier offline calculation on the same cases. Paired question-level percentile bootstrap intervals (20,000 resamples; seed 20261127) for the change are:

- Recall@5: +2.04 pp; 95% CI [+0.30, +3.80].
- Hit@5: +1.91 pp; 95% CI [+0.21, +3.62].
- MRR: -0.13 pp; 95% CI [-1.82, +1.50].

The bootstrap resamples questions as paired units. The intervals describe uncertainty over this benchmark's question sample, not generalization to a new dataset. MRR does not show a clear improvement.

## Recall@5 by question type

| Question type | Questions | Default BM25 | Stopword ablation | Change |
|---|---:|---:|---:|---:|
| knowledge-update | 72 | 81.48% | 80.79% | -0.69 pp |
| multi-session | 121 | 47.36% | 53.22% | +5.87 pp |
| single-session-assistant | 56 | 85.71% | 85.71% | 0.00 pp |
| single-session-preference | 30 | 33.33% | 36.67% | +3.33 pp |
| single-session-user | 64 | 89.84% | 90.62% | +0.78 pp |
| temporal-reasoning | 127 | 66.25% | 67.43% | +1.18 pp |

The largest apparent gain is multi-session retrieval, while knowledge-update regresses slightly. Small type-specific sample sizes, especially preference (30 questions), make these subgroup estimates noisy.

## Interpretation and limitations

This is now a reproducible, implemented ablation, rather than only an offline re-scoring exercise. It still remains exploratory: the stopword list was manually selected after inspecting errors on this same test split. The benchmark is therefore not an untouched validation set, and the apparent gains may be optimistic. The ablation should not replace default BM25 as the research control or be described as a general improvement.

Next evaluation should freeze the list and test it on a separately held-out dataset or a genuinely untouched split. Keep default BM25 as the control, report paired per-question changes and all question-type regressions, and do not tune the list against the held-out results. Turn-level Recall@K is evidence retrieval, not answer accuracy.
