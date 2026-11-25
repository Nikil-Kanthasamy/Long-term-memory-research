# Independent paraphrase-fact joint-span evaluation

- Workflow: [45680273756](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45680273756)
- Workflow conclusion: success
- Evaluated code commit: `ce66b3f1ef0790fdecd4a264253dee3a64dc8135`
- Fixture: `examples/tiny_lm_memory_qa_independent_paraphrase_eval.jsonl`
- Fixture size: 40 question rows over 20 new fact clusters, with two differently phrased questions per cluster
- Seeds: 13, 29, 47
- Training: 600 joint-span supervision steps per seed; CPU workflow
- Initialization: same copy-enabled base checkpoint initialization per seed; language model and joint-span head fine-tuned; independent span-head initialization
- Candidate protocol: all ordered spans with start <= end over retained memory bytes; no gold answer length or length-conditioned candidate windows
- Eligibility: all 40 questions had an answer span in retained memory for each seed; zero skipped

## Results

| Seed | Eligible cases | Skipped | Start top-1 | End top-1 | Exact span | Mean gold-span rank |
|---:|---:|---:|---:|---:|---:|---:|
| 13 | 40 | 0 | 50.00% | 40.00% | 14/40 (35.00%) | 57.78 |
| 29 | 40 | 0 | 45.00% | 40.00% | 10/40 (25.00%) | 57.98 |
| 47 | 40 | 0 | 35.00% | 42.50% | 10/40 (25.00%) | 63.88 |
| **Mean across seeds** | **40 per seed** | **0** | **43.33%** | **40.83%** | **28.33%** | **59.88** |

The three seed runs evaluate the same 40 rows. The 120 seed-case evaluations are repeated measurements over the same 20 fact clusters, not 120 independent questions. Exact-span accuracy ranged from 25% to 35% across seeds. Before joint-span supervision, exact-span accuracy was 0% for all three seeds; mean gold-span ranks before supervision were 250.98, 976.05, and 878.73, respectively.

Within each two-question paraphrase pair, outcomes were consistent for this fixture: seed 13 exactly localized both variants for 7 of 20 clusters, and seeds 29 and 47 did so for 5 of 20 clusters each; the remaining clusters had neither variant exactly localized. This is descriptive of this small constructed set, not evidence of universal paraphrase invariance.

## Interpretation

Joint-span supervision improved ranking and exact localization relative to the unsupervised checkpoint on this independent synthetic fixture, but the resulting exact-span accuracy was modest: 28.33% mean across seeds, with a 10 percentage-point range. The earlier reused eight-fact paraphrase fixture produced substantially higher scores, so that earlier result should not be treated as representative of performance on new facts. This independent test is a more sobering estimate and weakens any broad generalization claim.

The task is extractive span localization, not free-form answer generation. The fixture is synthetic and small; the three seeds measure optimization variability on the same examples. It does not establish real-world long-term-memory performance, semantic robustness to natural user paraphrases, or reliable verbalization of retrieved answers.

## Artifact

The workflow artifact `tiny-memory-lm-wikitext2-pilot-45680273756` contains per-seed JSON metrics, logs, and checkpoints under `independent-paraphrase-joint-span-seed-sweep/`. The aggregate metrics are in `joint-span-seed-summary.json`.
