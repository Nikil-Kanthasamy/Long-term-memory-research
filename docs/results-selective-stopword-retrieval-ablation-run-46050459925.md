# Selective stopword retrieval ablation across three benchmarks

**Evaluation date:** 27 November 2026  
**Implementation branch:** `research/long-term-memory-starter`  
**Status:** Exploratory; no variant promoted to the default retriever.

## Question

The broader frozen stopword list improved turn-level factual retrieval on LongMemEval-S and LoCoMo but regressed whole-session personalized-guidance retrieval on PRAGMA. This follow-up asks whether a much narrower removal policy can avoid deleting potentially useful temporal, relational, interrogative, possessive, and action cues.

## Frozen policy

The selective condition removes exactly these five terms from both query and document token streams:

```text
a, an, the, and, or
```

It preserves the remaining terms in the broader list, including `what`, `when`, `did`, `to`, `from`, `about`, `ago`, `have`, `has`, `had`, `how`, `many`, `which`, `who`, and `mentioned`.

The primary control remains unchanged Okapi BM25 (`k1=1.5`, `b=0.75`). The broad stopword condition is retained as a comparator. The selective condition is implemented as a separate research arm and does not change default retrieval behavior.

## Aggregate results

All deltas below are selective condition minus default BM25, in percentage points (pp).

| Benchmark | Eligible cases | Recall@5 | Hit@5 | MRR | Recall@10 | Hit@10 |
|---|---:|---:|---:|---:|---:|---:|
| LongMemEval-S turn retrieval | 470 | 67.15% → 67.47% (+0.32) | 79.36% → 79.57% (+0.21) | 66.30% → 66.44% (+0.14) | 76.23% → 76.82% (+0.59) | 84.89% → 85.53% (+0.64) |
| LoCoMo turn retrieval | 1,531 | 41.20% → 40.98% (-0.23) | 45.72% → 45.59% (-0.13) | 34.96% → 34.94% (-0.02) | 48.49% → 48.79% (+0.31) | 53.63% → 53.89% (+0.26) |
| PRAGMA session retrieval | 400 | 62.95% → 63.02% (+0.07) | 96.25% → 96.25% (0.00) | 84.16% → 84.16% (0.00) | 80.27% → 80.15% (-0.12) | 99.50% → 99.50% (0.00) |

## Paired uncertainty

The bootstrap uses 20,000 resamples and seed `20261127`. The bootstrap unit follows each benchmark's sampling structure.

### LongMemEval-S

Question-level paired bootstrap over 470 questions:

| Metric | Mean change | 95% CI |
|---|---:|---:|
| Recall@5 | +0.32 pp | [0.00, +0.85] |
| Hit@5 | +0.21 pp | [0.00, +0.64] |
| MRR | +0.14 pp | [-0.33, +0.59] |
| Recall@10 | +0.59 pp | [+0.05, +1.28] |
| Hit@10 | +0.64 pp | [0.00, +1.49] |

The gains are small and concentrated in very few questions: Recall@5 improved on 2 questions, regressed on 0, and was unchanged on 468; Hit@5 improved on 1, regressed on 0, and was unchanged on 469. MRR improved on 35, regressed on 27, and was unchanged on 408. This is not a broad per-question improvement.

Recall@5 changed by +0.41 pp for multi-session and +1.79 pp for single-session-assistant questions; the other reported question types were unchanged at this cutoff.

### LoCoMo

Conversation-cluster paired bootstrap across 10 conversations (retain all questions for each sampled conversation):

| Metric | Mean change | 95% CI |
|---|---:|---:|
| Recall@5 | -0.23 pp | [-0.53, +0.07] |
| Hit@5 | -0.13 pp | [-0.45, +0.18] |
| MRR | -0.02 pp | [-0.47, +0.46] |
| Recall@10 | +0.31 pp | [-0.48, +0.98] |
| Hit@10 | +0.26 pp | [-0.50, +0.91] |

All intervals include zero. Recall@5 changes by query category were: multi-hop 0.00 pp, open-domain -0.12 pp, single-hop -0.06 pp, and temporal -0.73 pp. The small number of conversation clusters limits the precision of this uncertainty estimate.

### PRAGMA

User-cluster paired bootstrap across 100 users (retain all of a sampled user's queries):

| Metric | Mean change | 95% CI |
|---|---:|---:|
| Recall@5 | +0.07 pp | [-0.17, +0.30] |
| Hit@5 | 0.00 pp | [0.00, 0.00] |
| MRR | 0.00 pp | [0.00, 0.00] |
| Recall@10 | -0.12 pp | [-0.32, 0.00] |
| Hit@10 | 0.00 pp | [0.00, 0.00] |

Recall@5 by query type changed by +0.60 pp for Event-Align, -0.34 pp for Event-Correct, and 0.00 pp for both Trajectory-Align and Trajectory-Correct. No aggregate improvement is established.

## Comparison with the broader stopword list

The broad list produced larger LongMemEval-S and LoCoMo gains, but its PRAGMA regression remains material:

- LongMemEval-S: broad-list Recall@5 69.19% and MRR 66.17%, versus selective 67.47% and 66.44%.
- LoCoMo: broad-list Recall@5 44.79% and MRR 38.32%, versus selective 40.98% and 34.94%.
- PRAGMA: broad-list Recall@5 61.90% and MRR 82.99%, versus selective 63.02% and 84.16%.

The selective policy largely avoids the broad-list regression on PRAGMA, but it also removes most of the apparent gains on the two factual-retrieval benchmarks. It does not show a consistent advantage across tasks.

## Limitations

1. These three benchmarks have already been inspected in earlier experiments. This follow-up is exploratory, not a new held-out validation.
2. LongMemEval confidence intervals are question-level; LoCoMo intervals are conversation-cluster-level with only ten clusters; PRAGMA intervals are user-cluster-level.
3. Retrieval relevance is based on benchmark evidence annotations. No answer generation, answer correctness, personalized-guidance quality, or user outcome is evaluated here.
4. A small difference in aggregate retrieval metrics should not be interpreted as a meaningful product improvement without downstream evaluation.
5. The term list is a simple lexical ablation, not a learned or adaptive query policy.

## Decision

**Keep default BM25 unchanged. Do not promote either stopword list as a universal improvement.** The selective five-term policy is closer to neutral across benchmarks, but its gains are small, inconsistent, and uncertain. The broad list remains task-dependent.

A stronger next direction is to investigate retrieval features that preserve the full query—such as temporal metadata, entity-aware matching, or hybrid lexical/dense ranking—using a development split for design and a genuinely untouched benchmark or user cohort for final validation.

## Reproducibility

- LongMemEval-S selective run: [workflow 46050459925](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/46050459925)
- LoCoMo selective run: [workflow 46050454288](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/46050454288)
- PRAGMA selective run: [workflow 46050454207](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/46050454207)
- Latest unit tests: [workflow 46050460084](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/46050460084)

The benchmark workflows download their datasets at run time and upload retrieval metrics only. PRAGMA histories and user profiles are not uploaded.
