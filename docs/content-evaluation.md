# Content-quality evaluation

## Goal

Compare drafts from the same brief under three conditions:

1. **no_memory** — memory retrieval is disabled.
2. **memory** — user-scoped long-term memories are supplied as reference context.
3. **memory_plus_review** — a memory-assisted draft receives a separate advisory editorial review. The draft is not edited by that review.

The harness does not assign quality scores automatically. It creates draft artifacts, a private provenance manifest, and a human-rating CSV template. Draft filenames use blind IDs only, and the rating rows are shuffled with a reproducible seed. Keep the manifest private because it maps blind IDs to conditions and memory provenance.

## Dataset

Provide a non-empty JSON array. Each case requires a topic; other fields are optional:

~~~json
[
  {
    "id": "brief-001",
    "topic": "Why long-term memory matters for AI assistants",
    "content_type": "article",
    "audience": "technology students",
    "tone": "clear and educational",
    "length": "medium",
    "keywords": ["retrieval", "personalization"],
    "instructions": "Use practical examples but do not invent research results."
  }
]
~~~

## Run

~~~
python -m src.ltm.cli content evaluate --user demo --dataset data/content_eval.json --output-dir results/content-eval
~~~

For a two-condition comparison, add `--no-review`. The default makes three draft-generation requests per case (one per condition) plus one separate editorial-review request for the memory-plus-review condition. API usage and cost scale with case count.

## Outputs

- `drafts/D0001.md`, etc.: generated Markdown drafts with blind IDs only.
- `manifest.jsonl`: blind draft ID, case, condition, draft path, memory count and memory IDs, plus advisory review where applicable. This file is for the study coordinator only.
- `ratings_template.csv`: shuffled blind draft IDs, reviewer ID field, and empty rating columns. Add duplicate rows for each additional reviewer.
- `summary.json`: experiment size, conditions, rating dimensions and randomization seed.

## Human rating rubric

Rate these dimensions from 1 (poor) to 5 (excellent): relevance to brief, personalization, factual consistency with supplied context, style consistency, completeness, and readability. For `unsupported_claims`, enter a count of distinct factual claims not supported by the brief or memory evidence. Flag citations, statistics, quotations, product specifications, and concrete factual assertions for verification.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Relevance | misses the brief | partly addresses it | directly fulfills it |
| Personalization | irrelevant or absent | some useful tailoring | relevant, specific tailoring |
| Factual consistency | major conflicts | mixed / needs checks | consistent with supplied evidence |
| Style consistency | poor fit | acceptable | strong fit for audience and tone |
| Completeness | major omissions | adequate | complete for requested format |
| Readability | difficult | understandable | clear and engaging |

## Analysis

Complete ratings independently, adding a distinct `reviewer_id` for each reviewer. Then run:

~~~
python scripts/analyze_content_eval.py --ratings results/content-eval/ratings_template.csv --manifest results/content-eval/manifest.jsonl --output results/content-eval/analysis.json
~~~

The script reports per-condition means and sample standard deviations for each dimension, missing ratings, matched case-level mean differences, and descriptive exact/within-one reviewer agreement where reviewer IDs are supplied. Multiple reviewers are averaged within each draft before paired comparisons.

Do not claim improvement unless the data supports it. The analysis does not calculate statistical significance, confidence intervals, or chance-corrected inter-rater reliability. Report sample size, rating rubric, reviewer count, missingness, and model/version.

## Pilot protocol

See [Content-quality pilot study](content-evaluation-pilot.md) for the eight-brief synthetic pilot, memory seeding, API-call budget, blinding protocol, and analysis workflow.

## Limitations and data handling

- Generation is nondeterministic; record model/version, date, and settings for publishable experiments.
- Editorial review is model-generated and advisory, not independent fact verification.
- Use synthetic/non-sensitive memories and a dedicated test user; the default memory-enabled conditions send retrieved context to Gemini.
- Review current Gemini data-handling terms before using hosted generation with private content.
