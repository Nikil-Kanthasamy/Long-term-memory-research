# Content-quality pilot study

## Purpose

This pilot makes the existing content-quality evaluation repeatable using synthetic, non-sensitive project context. It compares three conditions on matched briefs:

1. `no_memory`: generation without retrieved memories.
2. `memory`: generation with user-scoped retrieved memories.
3. `memory_plus_review`: a separate advisory review is generated for a memory-assisted draft. The review does **not** revise the draft.

The pilot dataset is deliberately small (8 briefs). It is a workflow and rubric pilot, not a statistically powered study and not evidence that one condition is better.

## Files

- `examples/content_eval_pilot.json`: eight content briefs.
- `examples/content_eval_pilot_memories.json`: synthetic memory records for a dedicated pilot user.
- `scripts/seed_content_eval_pilot.py`: inserts the synthetic memories into the configured SQLite database. Exact duplicate memory text is not duplicated.
- `scripts/analyze_content_eval.py`: analyzes completed ratings, condition summaries, paired case-level differences, and descriptive reviewer agreement.

## Run the pilot

Use a fresh test database or confirm `LTM_DB_PATH` points to a database where you are comfortable adding synthetic records. The script never deletes existing data.

```bash
python scripts/seed_content_eval_pilot.py --user content-eval-pilot
python -m src.ltm.cli content evaluate \
  --user content-eval-pilot \
  --dataset examples/content_eval_pilot.json \
  --output-dir results/content-eval-pilot
```

Default mode generates 24 drafts and makes 8 additional editorial-review requests: 32 hosted generation calls total, excluding optional embedding calls made by retrieval. To reduce API use, add `--no-review` for 16 drafts and no review calls. The content briefs and, in memory-enabled conditions, retrieved memory context are sent to the configured Gemini API. Use only the included synthetic data unless you have reviewed provider data-handling terms and obtained appropriate consent.

## Blinding and ratings

Draft filenames are blind IDs only (for example, `drafts/D0001.md`); they do not expose the condition. The ratings CSV order is shuffled reproducibly using seed 2026. Keep `manifest.jsonl` private: it reveals the condition and memory provenance. Do not share it with reviewers.

For each reviewer, duplicate the CSV rows and fill a distinct `reviewer_id`. Each reviewer should rate every draft independently. Rate relevance, personalization, factual consistency, style consistency, completeness, and readability from 1 (poor) to 5 (excellent). Enter a non-negative integer count for `unsupported_claims`. Leave a cell blank if the reviewer cannot rate that dimension; do not guess.

The existing rubric is qualitative. Reviewers should flag unsupported statistics, quotations, citations, specifications, and concrete factual assertions in reviewer notes. Generated drafts must be checked before any publication.

## Analyze ratings

After reviewers complete the CSV:

```bash
python scripts/analyze_content_eval.py \
  --ratings results/content-eval-pilot/ratings_template.csv \
  --manifest results/content-eval-pilot/manifest.jsonl \
  --output results/content-eval-pilot/analysis.json
```

The analysis reports per-condition means and sample standard deviations, missing values, matched case-level mean differences (memory minus no-memory; memory-plus-review minus memory), and descriptive exact/within-one reviewer agreement when reviewer IDs are supplied. Multiple reviewers are averaged within each draft before paired comparisons.

The agreement figures are descriptive only, not chance-corrected inter-rater reliability. The script does not calculate significance tests, confidence intervals, or causal effects. With only eight cases, treat all results as exploratory. Report sample size, reviewer count, missingness, model/version, commit, date, prompts/settings, token use, latency, and cost alongside results.

## Important interpretation

- The three conditions are not identical in API-call count: the review condition has an additional review request.
- `memory_plus_review` evaluates a review artifact, not a draft revised using that review.
- The model is nondeterministic; preserve generated drafts and metadata.
- The pilot is synthetic and must not be described as a user study.
- No quality scores or improvement claims exist until real human ratings are collected and analyzed.
