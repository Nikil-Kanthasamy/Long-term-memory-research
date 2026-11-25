# Paraphrased-question contextual joint-span evaluation

- Workflow: [45529718382](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45529718382)
- Workflow conclusion: success
- Evaluated code commit: `a74737b9f83a357ddbc4a2c2ca4c7577ccaa87b1`
- Fixture: `examples/tiny_lm_memory_qa_paraphrase_alias_eval.jsonl`
- Seeds: 13, 29, 47
- Training: 600 steps per seed, batch size 4, CPU
- Initialization: same copy-enabled base checkpoint per seed; both language model and joint-span head fine-tuned; independent span-head initialization
- Candidate protocol: all ordered spans with start <= end over retained memory bytes; no gold answer length or length-conditioned candidate windows

## Results

| Seed | Eligible cases | Skipped missing answer span | Start top-1 | End top-1 | Exact span | Mean gold-span rank |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 14 | 2 | 71.43% | 64.29% | 9/14 (64.29%) | 36.57 |
| 29 | 14 | 2 | 57.14% | 78.57% | 10/14 (71.43%) | 53.86 |
| 47 | 14 | 2 | 57.14% | 50.00% | 10/14 (71.43%) | 45.36 |
| **Pooled** | **42 seed-case evaluations** | **6 seed-case skips** | — | — | **29/42 (69.05%)** | **45.26** |

The fixture contains 16 question rows arranged as paired paraphrases over 8 fact clusters. Fourteen rows have a canonical answer span available in the retained memory; two rows are skipped for that reason in each seed. The 42 seed-case evaluations are repeated runs over the same 14 eligible rows, not 42 independent examples.

## Interpretation

The contextual joint-span selector localized the canonical answer text for 29 of 42 seed-case evaluations (69.05%) without being given the gold answer length. Each seed achieved between 64.29% and 71.43% exact-span accuracy. This is evidence that the selector can recover a canonical extractive span under the fixture's paraphrased questions.

This is a span-localization result, not a free-form answer-generation result and not evidence that the small language model can reliably verbalize the answer. It also does not establish broad paraphrase generalization: the fixture is small, uses paired paraphrases of only eight fact clusters, and reuses the existing evaluation fixture rather than an independently collected test set. The three seeds measure training variability on the same examples.

The earlier contextual selector replication reported 47.95% mean exact-span accuracy on a different 57-case unseen-fact evaluation. That percentage is not a like-for-like comparison with this 14-case eligible paraphrase subset; the datasets and question distributions differ.

## Artifacts

The workflow artifact `tiny-memory-lm-wikitext2-pilot-45529718382` contains per-seed JSON metrics, logs, and checkpoints under `paraphrase-joint-span-seed-sweep/`. The seed summary is `joint-span-seed-summary.json`.
