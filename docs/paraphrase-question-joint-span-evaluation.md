# Joint-span selection on paraphrased questions

## Question

The alias-aware generation evaluation found retrieval evidence in all 16 cases but zero accepted generated answers. This follow-up tests whether the contextual answer-length-free joint-span selector can still locate the canonical answer span when the *question wording* is paraphrased.

## Protocol

The full CPU workflow trains three independent contextual joint-span selectors (seeds 13, 29, 47) from the same copy-enabled base checkpoint, using the same extractive training set and 600-step settings as the earlier replication. Evaluation uses `examples/tiny_lm_memory_qa_paraphrase_alias_eval.jsonl` (16 rows / 8 fact clusters). The selector scores all ordered start/end spans over retained memory bytes without receiving the gold answer length.

The evaluator's target is each row's canonical `expected_answer`, which is present in the corresponding memory text. Curated aliases are **not** used to expand the target span. This is therefore a test of locating an extractive canonical answer under paraphrased questions, not a test of semantic answer generation or alias matching.

## Workflow

The `tiny LM model tests` workflow runs `scripts/run_tiny_lm_joint_span_seeds.py` against the alias fixture and uploads the three per-seed checkpoints, logs, evaluations, and summary in `paraphrase-joint-span-seed-sweep/`.

## Metrics and caveats

Report per-seed and mean exact-span accuracy, start/end top-1 accuracy, mean joint-span rank, and skipped cases. Cases where the canonical answer does not appear in retained memory are skipped by the existing evaluator and must be reported.

This fixture reuses the same eight fact clusters as the alias-aware generation evaluation and is not an independent generalization benchmark. With only eight independent fact clusters, results are exploratory. Do not compare its raw accuracy directly with the 57-case unseen-facts benchmark without noting the different case set and size.
