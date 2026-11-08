# Memory-aware multi-format content creation

## Purpose

The content creation module turns the existing long-term-memory prototype into a multi-purpose drafting assistant. It accepts a structured brief, retrieves relevant user-scoped memories, places only those memories into the generation prompt, and asks the configured Gemini model to create a draft. It supports blogs, articles, social posts, emails, newsletters, reports, scripts, and product descriptions.

This is an application-level generation pipeline using a pretrained hosted model. It does not train a new foundation model or fine-tune Gemini.

## Architecture

~~~text
User brief
  | topic, format, audience, tone, length, keywords, instructions
  v
CLI: content create
  |
  +--> SQLite memory store --> user-scoped retrieval --> relevant memory snippets
  |                                                   |
  +---------------------------------------------------+
                                                      v
                                         Prompt builder + format guidance
                                                      |
                                                      v
                                         Gemini generation (LLM API)
                                                      |
                                                      v
                                                Draft output
                                                      |
                                  +-------------------+------------------+
                                  |                                      |
                                  v                                      v
                           Markdown/text file                   Optional editorial review

Existing draft --> content revise --> Gemini revision --> revised draft
~~~

## Supported formats

| Format key | Intended output |
|---|---|
| blog | Title, sections, examples, conclusion |
| article | Informative, structured article |
| social | Hook, short copy, optional call to action |
| email | Subject, greeting, body, sign-off |
| newsletter | Subject, intro, scannable sections, closing |
| report | Summary, background, analysis, recommendations, conclusion |
| script | Spoken opening, transitions, speaker-friendly text |
| product | Benefits-led description without invented specifications |

## Quick start

Configure GEMINI_API_KEY in .env as described in the main README. The content generator requires the key because it calls the configured Gemini text-generation model. Memory retrieval can use semantic embeddings when configured and falls back to lexical retrieval if embeddings are unavailable.

Generate a blog post:

~~~bash
python -m src.ltm.cli content create \
  --user demo \
  --topic "Why long-term memory matters for AI assistants" \
  --type blog \
  --audience "technology students" \
  --tone "clear and educational" \
  --length medium \
  --keywords memory RAG retrieval \
  --output results/ai-memory-blog.md
~~~

Generate a report with an optional separate editorial review:

~~~bash
python -m src.ltm.cli content create \
  --user demo \
  --topic "Evaluation of a conversational memory prototype" \
  --type report \
  --audience "research supervisors" \
  --tone formal \
  --length long \
  --review \
  --output results/memory-report.md
~~~

Revise an existing Markdown or text draft:

~~~bash
python -m src.ltm.cli content revise \
  --input results/ai-memory-blog.md \
  --instructions "Make the introduction shorter and add a practical example." \
  --output results/ai-memory-blog-revised.md
~~~

Run the unit tests:

~~~bash
python -m unittest discover -s tests -v
~~~

## Design decisions

1. Memory is optional context. If no relevant memories are found, the model still receives the brief and can create a draft.
2. User scoping is preserved. Retrieval calls the existing memory retrieval function with the supplied user ID.
3. No automatic memory writes. Generated drafts are not automatically saved as user facts. This avoids polluting factual memory with generated or unverified prose.
4. Traceability. The command reports the IDs of memories used for the draft.
5. Separate revision. Revision accepts an explicit instruction and a source file; it does not silently overwrite the source.
6. Editorial review is advisory. The optional review checks clarity, structure, audience fit, tone consistency, factual risk, and improvements. It is not a fact-checking service.
7. Grounded generation. The prompt tells the model not to invent statistics, quotations, citations, product specifications, or unsupported claims. Missing critical facts may be marked with [VERIFY: ...].

## Current limitations

- Generation depends on the configured Gemini API and network availability.
- Quality review is model-generated and should be treated as editorial feedback, not an objective score.
- The module does not yet provide a graphical editor, web research/citation retrieval, image generation, publishing integrations, or a dedicated fine-tuned model.
- Generated facts and citations must be checked by a human before publication, especially for academic, legal, medical, financial, or product claims.
- Review API cost and latency are additional when --review is enabled.

## Research evaluation plan

To study whether long-term memory improves content generation, compare at least two conditions using the same briefs and generation model:

- No-memory baseline: generate from the brief alone.
- Memory-assisted condition: retrieve user-scoped memories and include relevant evidence.

Evaluate blind-rated relevance, personalization, factual consistency with supplied context, style consistency, completeness, readability, and unsupported-claim rate. Use the same model/version, temperature, and prompt structure where possible. Report sample size, rating rubric, and inter-rater agreement. Do not treat retrieval metrics such as Precision@K or MRR as a substitute for content-quality evaluation.
