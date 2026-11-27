# LongMemEval-S turn retrieval failure analysis — run 45926510628

## Reproduction

- Workflow: [45926510628](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45926510628)
- Branch: `research/long-term-memory-starter`
- Workflow result: all 10 steps passed.
- Artifact: `longmemeval-s-retrieval-comparison-45926510628` (artifact ID 15166004205; retained by GitHub for 30 days).
- Dataset: official cleaned LongMemEval-S; 500 questions, 470 answerable and 30 abstentions.
- Coverage: 470/470 answerable questions (100%); no skipped questions.
- Corpus: 231,595 nonempty turns and 886 labeled evidence turns.
- Relevance: only official turn-level `has_answer=true` labels.
- Scope: retrieval of evidence turns only, not answer generation or answer correctness.

## BM25 turn-level results

| Metric | BM25 |
|---|---:|
| Precision@1 | 55.32% |
| Recall@1 | 37.13% |
| Hit@1 | 55.32% |
| Precision@3 | 31.84% |
| Recall@3 | 58.06% |
| Hit@3 | 74.47% |
| Precision@5 | 22.77% |
| Recall@5 | 67.15% |
| Hit@5 | 79.36% |
| Precision@10 | 13.21% |
| Recall@10 | 76.23% |
| Hit@10 | 84.89% |
| MRR | 66.30% |

The first labeled evidence turn appeared at rank 1 for 260 questions, ranks 2–5 for 113, ranks 6–10 for 26, and below rank 10 for 71. No answerable question lacked a labeled turn in the complete ranking. Thus, the 15.11% with no Hit@10 are ranking-depth misses, not missing evidence coverage.

## Per-question-type BM25 results

| Question type | Questions | Recall@5 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| knowledge-update | 72 | 81.48% | 90.28% | 74.48% |
| multi-session | 121 | 47.36% | 76.03% | 59.00% |
| single-session-assistant | 56 | 85.71% | 85.71% | 78.90% |
| single-session-preference | 30 | 33.33% | 40.00% | 22.07% |
| single-session-user | 64 | 89.84% | 90.62% | 80.97% |
| temporal-reasoning | 127 | 66.25% | 77.17% | 66.12% |

## What the hardest examples show

The artifact includes the 25 questions with the worst first-evidence rank, alongside the top five retrieved turns and the annotated evidence turns. The sample contains 10 temporal-reasoning, 7 preference, 4 multi-session, 3 assistant-session, and 1 user-session questions.

Representative failure patterns:

1. **Conceptual paraphrase / preference retrieval.** The question asks for advice on choosing a guitar for a music-store visit; its labeled evidence discusses upgrading from a Fender Stratocaster to a Gibson Les Paul. The shared word “guitar” is not enough to distinguish the evidence from many other guitar-related turns.
2. **Temporal phrasing.** Questions use relative expressions such as “four weeks ago,” “two weeks ago,” and “5 days ago,” while the evidence turn describes an event in ordinary conversational language (“today,” “yesterday,” or an activity description). Plain BM25 does not resolve the temporal relation.
3. **Multi-session aggregation.** Questions asking how many places were visited or the chronological order of several events require finding multiple evidence turns, not just one semantically similar turn. A single-query BM25 ranking can retrieve only a subset near the top.

The examples are diagnostic, not a representative random sample; they were deliberately selected by worst first-evidence rank.

## Recommended next experiment

Keep default BM25 (Okapi k1=1.5, b=0.75) as the fixed control. Evaluate two isolated additions against the same 470 questions:

- **Query normalization / paraphrase handling:** compare a deterministic stemming or synonym-expansion variant against the unchanged tokenizer. Report overall and per-type Recall@5, Hit@5, MRR, plus paired per-question deltas. Do not treat a tuned-on-test result as a held-out estimate.
- **Temporal candidate scoring:** separately test a conservative date-aware prior only where a question contains an explicit relative-time expression and the benchmark provides an unambiguous reference date. If the reference date cannot be established from the benchmark record, abstain from applying the prior rather than guessing.

For both experiments, retain the original BM25 rank as a baseline and report regressions by question type. Do not combine both changes in the first ablation; otherwise gains cannot be attributed.

## Limitations

- Evidence relevance is the benchmark's binary turn annotation; semantic sufficiency is not independently judged.
- Turn Recall@K is not directly comparable with session-level Recall@K because the retrieval units and relevance labels differ.
- The 25 worst examples are an error-inspection set, not a statistically representative sample.
- No reader or answer-generation model was evaluated in this run.


## Exploratory offline ablation: remove common words

After downloading the complete run artifact, the same 470 question cases were re-scored locally with an exploratory tokenizer variant. The variant removed this fixed list from both query and turn text before BM25: `what, was, the, i, my, a, an, is, did, do, of, in, on, to, for, me, you, and, or, it, that, one, how, many, which, who, when, where, from, with, about, ago, have, has, had, mentioned, tell, can`. All other BM25 settings remained k1=1.5 and b=0.75. The unmodified tokenizer reproduced the artifact metrics exactly.

| Metric | Default BM25 | Remove common words | Paired change |
|---|---:|---:|---:|
| Recall@5 | 67.15% | 69.19% | +2.04 pp |
| Hit@5 | 79.36% | 81.28% | +1.91 pp |
| MRR | 66.30% | 66.17% | -0.13 pp |

A paired question-level percentile bootstrap (20,000 resamples; seed 20261127) gave these 95% intervals for the change:

- Recall@5: +0.30 to +3.80 percentage points.
- Hit@5: +0.21 to +3.62 percentage points.
- MRR: -1.82 to +1.50 percentage points.

Per-type Recall@5 changed as follows:

| Question type | Default | Remove common words |
|---|---:|---:|
| knowledge-update | 81.48% | 80.79% |
| multi-session | 47.36% | 53.22% |
| single-session-assistant | 85.71% | 85.71% |
| single-session-preference | 33.33% | 36.67% |
| single-session-user | 89.84% | 90.62% |
| temporal-reasoning | 66.25% | 67.43% |

This suggests the largest gains are in multi-session retrieval, with smaller gains for preference questions, while knowledge-update retrieval regresses slightly. The ablation is exploratory and was measured on the same benchmark split used to inspect failures; the word list was manually selected, and no held-out split was used. It is not evidence of a generalizable improvement and should not replace the primary BM25 baseline. Next: freeze the list, evaluate on a separate validation dataset or a clearly separated development split, and report per-type regressions before considering adoption.
