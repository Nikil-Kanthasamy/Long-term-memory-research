# PRAGMA session-retrieval evaluation: the frozen stopword ablation regresses

## Reproduction and provenance

- Benchmark workflow: [45930242216](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45930242216) (all steps passed).
- Unit tests: [45930242246](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45930242246) (passed).
- Branch: `research/long-term-memory-starter`.
- Benchmark code revision: `68fa96d63d06889bb46d2af2de2a66025c0669b9`.
- Dataset: `stellahj/PRAGMA`, resolved immutable Hugging Face revision `e6749080e437a6c83444cc0b239a28ff587a993e`.
- License file SHA-256: `d500614feef74e1e2cc91aa6cddd0c7b9098e79a5af45c92a6ab7455b5a99677`. The dataset card specifies the NVIDIA License and non-commercial research/evaluation use. Source files are downloaded at CI runtime; only metrics are uploaded.
- Dataset scale: 100 users, 400 queries (100 per query type), 4,167 timestamped sessions (3,000 filler, 494 event, 673 trajectory).
- All 400 queries were eligible. No missing evidence, unknown users, invalid indices, empty histories, or future-evidence exclusions were observed.

PRAGMA evaluates personalized guidance over evolving user histories. It is distinct from LongMemEval and LoCoMo, and it measures retrieval of evidence **sessions**, not dialogue turns.

## Protocol

For each query, only sessions belonging to that user and timestamped at or before the query timestamp are searchable. The benchmark's `evidence_session_index` values identify the gold evidence sessions. Each whole session is one retrieval unit; its role/content messages are concatenated, with session type and timestamp included.

- **Control:** unchanged Okapi BM25, `k1=1.5`, `b=0.75`.
- **Ablation:** same BM25 implementation and parameters with the exact frozen LongMemEval stopword list. No PRAGMA-specific tuning.
- **Comparator:** token-set Jaccard.
- **Cutoffs:** 1, 3, 5, and 10 sessions.
- **Uncertainty:** paired bootstrap by user, 20,000 resamples, seed 20261127. All four query types for a sampled user remain together.

This is evidence-session retrieval only. No response generation, personalized-guidance alignment scoring, or LLM judge was run.

## Aggregate results

| Metric | Jaccard | Default BM25 | Frozen stopword ablation | Change vs BM25 |
|---|---:|---:|---:|---:|
| Recall@5 | 58.01% | 62.95% | 61.90% | -1.05 pp |
| Hit@5 | 97.75% | 96.25% | 96.25% | 0.00 pp |
| MRR | 79.27% | 84.16% | 82.99% | -1.17 pp |
| Recall@10 | 90.04% | 80.27% | 78.76% | -1.51 pp |
| Hit@10 | 100.00% | 99.50% | 99.50% | 0.00 pp |

The paired user-cluster bootstrap intervals for ablation-minus-default were:

| Metric | Mean change | User-cluster bootstrap 95% CI |
|---|---:|---:|
| Recall@5 | -1.05 pp | [-1.64, -0.45] pp |
| Hit@5 | 0.00 pp | [-0.75, +0.75] pp |
| MRR | -1.17 pp | [-1.93, -0.52] pp |
| Recall@10 | -1.51 pp | [-2.26, -0.76] pp |
| Hit@10 | 0.00 pp | [0.00, 0.00] pp |

The intervals for Recall@5, MRR, and Recall@10 are wholly below zero. The ablation is not a neutral simplification on this benchmark: it causes a measurable loss in how much evidence is retrieved and how highly it is ranked.

## Results by query type

| Query type | Questions | Default BM25 Recall@5 | Ablation Recall@5 | Change |
|---|---:|---:|---:|---:|
| Event-Align | 100 | 56.00% | 55.00% | -1.00 pp |
| Event-Correct | 100 | 89.97% | 89.43% | -0.53 pp |
| Trajectory-Align | 100 | 50.76% | 48.55% | -2.21 pp |
| Trajectory-Correct | 100 | 55.08% | 54.63% | -0.45 pp |

MRR also declined in every type: Event-Align -1.10 pp, Event-Correct -0.50 pp, Trajectory-Align -2.53 pp, and Trajectory-Correct -0.56 pp. Trajectory-Align is the largest subgroup regression.

## Paired question outcomes

| Metric | Improved | Regressed | Unchanged |
|---|---:|---:|---:|
| Recall@5 | 6 | 31 | 363 |
| Hit@5 | 1 | 1 | 398 |
| MRR | 5 | 16 | 379 |
| Recall@10 | 14 | 47 | 339 |
| Hit@10 | 0 | 0 | 400 |

Question-level outcomes are paired within the same user histories; user-cluster bootstrap is the inferential unit.

## Interpretation across benchmarks

The fixed stopword list improved turn-evidence retrieval on LongMemEval-S and LoCoMo, but it regressed session-evidence retrieval on PRAGMA. The unit and task differ: PRAGMA retrieves complete sessions for personalized event/trajectory guidance, while the earlier evaluations retrieved dialogue turns for factual QA. These findings therefore do not support a universal stopword-removal rule.

A plausible explanation is that words removed by the fixed list sometimes encode useful relational or temporal cues in personalized guidance questions. This is a hypothesis, not established causal analysis; a separate ablation would be needed to identify which terms drive the regression.

## Research decision

**Do not promote the stopword ablation to the default BM25 implementation.** Keep the existing default unchanged. Treat stopword removal as a task-dependent experimental variant: it helped two factual-retrieval benchmarks but harmed all four PRAGMA query types. Any future adaptive or query-class-specific stopword policy must be designed on training/development data and validated on a new held-out user cohort, not tuned against these 400 PRAGMA questions.

The next useful experiment is a controlled query-feature ablation (for example, preserve temporal/relational terms while removing only generic function words), with the term groups fixed before evaluation and user-level paired uncertainty retained.
