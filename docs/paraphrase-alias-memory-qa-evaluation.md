# Curated paraphrase-alias evaluation for memory QA

## Purpose

The existing QA evaluator required one contiguous normalized form of `expected_answer`. This unfairly marked a generated paraphrase wrong even when it was an explicitly accepted wording. The evaluator now supports an optional `accepted_answers` list on each case.

- `expected_answer` remains the canonical reference used for teacher-forced likelihood scoring.
- `accepted_answers` must be a non-empty list of non-empty strings and must include the canonical answer after normalization.
- A generated answer is counted as matched when it contains any accepted phrase as a complete contiguous normalized token sequence.
- Token-set F1 is calculated against the best-scoring accepted phrase.
- Retrieval evidence is considered a hit when it contains any accepted phrase.

Cases without `accepted_answers` preserve the old behavior by implicitly using `[expected_answer]`.

## New fixture

`examples/tiny_lm_memory_qa_paraphrase_alias_eval.jsonl` contains 16 cases across 8 fact clusters, with two question paraphrases per cluster. Each case has a canonical answer and manually curated alternative phrasings. The underlying memory text remains the source of truth.

The fixture deliberately reuses eight fact clusters already present in the unseen-facts benchmark. It is therefore a scoring/protocol extension and a diagnostic slice, **not an independent test set** and not evidence of generalization to new facts. The workflow evaluates byte, BPE, and copy-enabled instruction-tuned checkpoints on it and uploads the outputs with the existing pilot artifact.

## Interpretation

This remains deterministic lexical evaluation, not semantic judging. It accepts only the listed phrasings (or their normalized forms); a valid but unlisted paraphrase can still be marked wrong. Curated aliases can also be too permissive if poorly authored, so they should be reviewed against the memory and question. Token-set F1 is a partial-overlap diagnostic, not a semantic similarity score.

The benchmark separates:
1. generated-answer acceptance against the curated rubric;
2. retrieved-evidence presence against the same rubric; and
3. teacher-forced canonical-answer likelihood.

These are distinct measurements. A retrieval hit or lower teacher-forced NLL does not establish correct free-form generation.

## Reproduction

Run:

```bash
python -m unittest tests.test_tiny_lm -v
python scripts/evaluate_tiny_lm_memory_qa.py \
  --checkpoint models/tiny-memory-lm-copy/instruct.pt \
  --cases examples/tiny_lm_memory_qa_paraphrase_alias_eval.jsonl \
  --output paraphrase-alias-evaluation.json \
  --max-new-tokens 32
```
