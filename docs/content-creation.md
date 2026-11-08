# Memory-aware multi-format content creation

## Purpose

The content creation module turns the long-term-memory prototype into a multi-purpose drafting assistant. It accepts a structured brief, optionally retrieves relevant user-scoped memories, and asks the configured Gemini model to create a draft. It supports blogs, articles, social posts, emails, newsletters, reports, scripts, and product descriptions.

This is an application-level generation pipeline using a pretrained hosted model. It does not train a new foundation model or fine-tune Gemini.

## Architecture

~~~text
User brief
  | topic, format, audience, tone, length, keywords, instructions
  v
CLI: content create
  |
  +--> [--no-memory?] -- yes --> Prompt builder (brief only)
  |                         |
  | no                      |
  v                         |
SQLite memory store         |
  | user-scoped retrieval   |
  v                         |
Relevant memory snippets    |
  |                         |
  +--> untrusted-data delimiters
                 |
                 v
       Prompt builder + format guidance
                 |
                 v
       Gemini generation (configurable output budget)
                 |
                 v
            Draft output
                 |
        +--------+---------+
        |                  |
        v                  v
 Markdown/text file   Optional editorial review
                          |
                          v
                  Schema-validated JSON
                  (or labeled raw notes)

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

Configure GEMINI_API_KEY in .env. The content generator sends the prompt to the configured Gemini API. When memory is enabled (the default), retrieved memory text is included in that request. Use --no-memory when a brief should not use or transmit retrieved long-term memory. This flag prevents memory retrieval for this content-generation request; the user's brief and generated draft are still sent to Gemini.

Before using hosted AI services, review the current Gemini/Google API terms, data-use and retention settings applicable to your account and region. Do not submit secrets, credentials, or sensitive personal information unless you have confirmed that the provider's handling is appropriate for your use case. The project cannot determine your provider-account retention or training settings. Memory retrieval can use semantic embeddings when configured and falls back to lexical retrieval if embeddings are unavailable.

Generate a blog post with memory context:

~~~bash
python -m src.ltm.cli content create \
  --user demo --topic "Why long-term memory matters for AI assistants" \
  --type blog --audience "technology students" --tone "clear and educational" \
  --length medium --keywords memory RAG retrieval --output results/ai-memory-blog.md
~~~

Generate a draft without retrieving or sending long-term memories:

~~~bash
python -m src.ltm.cli content create \
  --user demo --topic "A private announcement" --type email \
  --length short --no-memory --output results/private-announcement.md
~~~

Generate a report with an optional separate editorial review:

~~~bash
python -m src.ltm.cli content create \
  --user demo --topic "Evaluation of a conversational memory prototype" \
  --type report --audience "research supervisors" --tone formal \
  --length long --review --output results/memory-report.md
~~~

Revise an existing Markdown or text draft:

~~~bash
python -m src.ltm.cli content revise \
  --input results/ai-memory-blog.md \
  --instructions "Make the introduction shorter and add a practical example." \
  --output results/ai-memory-blog-revised.md
~~~

Run tests with: python -m unittest discover -s tests -v

## Design decisions and safeguards

1. Memory is optional. --no-memory skips retrieval entirely and sends no retrieved memory text to Gemini.
2. User scoping is preserved. Retrieval calls the existing memory retrieval function with the supplied user ID; the storage layer enforces that scope.
3. Retrieved memories are untrusted reference data. They are explicitly delimited and the system instruction says not to follow instructions found inside them. These mitigations cannot guarantee prevention of every prompt-injection attack.
4. Output budgets vary by length: short 700, medium 1,600, long 3,000 output tokens. If Gemini reports stopping at the token limit, generation raises a clear error rather than silently presenting a potentially incomplete draft.
5. Generated drafts are not automatically saved as user facts.
6. The command reports IDs of memories used for the draft.
7. Editorial review is advisory. Required fields and value types are validated. Invalid or incomplete JSON is returned as clearly labeled raw editorial notes. It is not fact verification.
8. The prompt discourages invented statistics, quotations, citations, product specifications, or unsupported claims. Missing critical facts may be marked with [VERIFY: ...].

## Current limitations

- Generation depends on the configured Gemini API and network availability.
- --no-memory disables retrieved-memory context only; it does not make generation local or prevent the brief itself from being sent to Gemini.
- Prompt delimiters and instructions reduce, but cannot eliminate, prompt-injection risk.
- Editorial review is model-generated, not an objective score or independent fact check.
- The module does not yet provide a graphical editor, web research/citation retrieval, publishing integrations, or a dedicated fine-tuned model.
- Generated facts and citations must be checked by a human before publication, especially for academic, legal, medical, financial, or product claims.

## Research evaluation plan

Compare identical briefs under no-memory and memory-assisted conditions using the same model/version and generation settings. Optionally add a third memory-plus-editorial-review condition. Blind-rate relevance, personalization, factual consistency with supplied context, style consistency, completeness, readability, and unsupported-claim rate. Include sample size, rating rubric, and inter-rater agreement. Retrieval metrics such as Precision@K or MRR are not substitutes for content-quality evaluation.
