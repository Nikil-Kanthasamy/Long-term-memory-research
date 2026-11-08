# Content-quality evaluation

## Goal

Compare drafts from the same brief under three conditions:

1. **no_memory** — memory retrieval is disabled.
2. **memory** — user-scoped long-term memories are supplied as reference context.
3. **memory_plus_review** — a memory-assisted draft receives a separate advisory editorial review. The draft is not edited by that review.

The harness does not assign quality scores automatically. It creates draft artifacts, a provenance manifest, and a human-rating CSV template.

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

~~~bash
python -m src.ltm.cli content evaluate --user demo --dataset data/content_eval.json --output-dir results/content-eval
~~~

For a two-condition comparison, add `--no-review`. The default makes two generation requests per case plus one separate review request for the third condition. API usage and cost scale with case count.

## Outputs

- `drafts/`: generated Markdown drafts.
- `manifest.jsonl`: case, condition, draft path, memory count and memory IDs, plus advisory review where applicable.
- `ratings_template.csv`: blind draft IDs and empty rating columns.
- `summary.json`: experiment size and rating dimensions.

The blind IDs are sequential in this starter template. A study coordinator should randomize presentation order and keep the manifest mapping away from reviewers; this tool does not implement full randomization.

## Human rating rubric

Rate these dimensions from 1 (poor) to 5 (excellent): relevance to brief, personalization, factual consistency with supplied evidence, style consistency, completeness, and readability. For `unsupported_claims`, enter a count of distinct factual claims not supported by the brief or memory evidence. Flag citations, statistics, quotations, product specifications, and concrete factual assertions for verification.

| Dimension | 1 | 3 | 5 |
|---|---|---|---|
| Relevance | misses the brief | partly addresses it | directly fulfills it |
| Personalization | irrelevant or absent | some useful tailoring | relevant, specific tailoring |
| Factual consistency | major conflicts | mixed / needs checks | consistent with supplied evidence |
| Style consistency | poor fit | acceptable | strong fit for audience and tone |
| Completeness | major omissions | adequate | complete for requested format |
| Readability | difficult | understandable | clear and engaging |

## Analysis

Compare paired drafts from the same case. Report per-condition means and dispersion for the six 1–5 dimensions, mean unsupported-claim counts, sample size, missing ratings, reviewer count, rubric, and inter-rater agreement. Do not claim improvement unless the data supports it. This starter does not calculate significance or inter-rater agreement.

The third condition measures the addition of an editorial review artifact, not the impact of applying its recommendations. A future fourth condition can revise the draft using review recommendations and rate that revised draft separately.

## Limitations and data handling

- Generation is nondeterministic; record model/version, date, and settings for publishable experiments.
- Editorial review is model-generated and advisory, not independent fact verification.
- Use synthetic/non-sensitive memories and a dedicated test user; the default memory-enabled conditions send retrieved context to Gemini.
- Review current Gemini data-handling terms before using hosted generation with private content.
