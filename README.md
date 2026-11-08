# Long-Term Memory Research

A Python research sandbox for persistent memory in language-model applications. It runs alongside a separate ML training pipeline and does not modify model weights.

## Baseline
- One content-generation model: Google Gemini API (gemini-2.5-flash by default) for generation, memory extraction, and consolidation.
- SQLite persistence for conversation turns and extracted memory records.
- Optional Gemini embeddings (gemini-embedding-001) with cosine similarity; keyword fallback works without an API key.
- Memory lifecycle: extract, retrieve, deduplicate, supersede, version, inspect history, and delete.
- Research notes, experiments, and evaluation metrics.

This is a working baseline, not a newly trained neural memory model. It combines a pretrained API model with an external memory store. Train a dedicated memory model only after measuring this baseline.

## Quick start
Python 3.10+ recommended.

~~~bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add GEMINI_API_KEY from https://aistudio.google.com/apikey to .env
python -m src.ltm.cli add --user demo --text "I am training a vision model with PyTorch."
python -m src.ltm.cli ask --user demo --question "What framework am I using?"
python -m src.ltm.cli generate --user demo --prompt "Suggest a memory-aware experiment plan."
~~~

Without an API key, add and keyword search still work; extraction and generation require the key.

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| GEMINI_API_KEY | unset | Google AI Studio API key |
| GEMINI_MODEL | gemini-2.5-flash | Generation and extraction model |
| GEMINI_EMBEDDING_MODEL | gemini-embedding-001 | Semantic embedding model |
| LTM_DB_PATH | data/memory.sqlite3 | SQLite database |
| LTM_TOP_K | 5 | Retrieved memories |

Check current model availability, free-tier quotas, and data-use terms before sending private content to a hosted API. Never commit keys or real memory data.

## Architecture
~~~text
Conversation -> raw turn log (SQLite)
                    |
                    v
             Gemini fact extraction
                    |
                    v
           deduplicate / consolidate
                    |
                    v
           memory records + vectors
                    |
Question -> semantic / keyword retrieval
                    |
                    v
            prompt with evidence
                    |
                    v
             Gemini generation
~~~

See docs/architecture.md, docs/research-plan.md, and docs/reading-list.md.

## CLI
- add --user ID --text TEXT: persist a turn and extract memories.
- search --user ID --query TEXT: inspect relevant memories.
- ask --user ID --question TEXT: answer using retrieved context.
- generate --user ID --prompt TEXT: generate content using memory context.
- forget --user ID --memory-id ID: permanently delete one memory.

## Tests
~~~bash
python -m unittest discover -s tests -v
~~~

Memory is distinct from fine-tuning: it stores task-specific information outside model weights and retrieves it at inference time. Keep these pipelines separate initially. Only export curated, consented examples to training after data-governance review.

## Phase 2: temporal and versioned memory

The advanced engine adds additive database migrations, explicit supersession of outdated facts, validity windows, raw conversation search, and timeline inspection. Existing Phase 1 SQLite databases are migrated automatically when the store initializes.

~~~bash
python -m src.ltm.cli history --user demo --query PyTorch
python -m src.ltm.cli timeline --user demo --memory-id 1
~~~

A correction can supersede a prior fact only when the extractor identifies its active memory ID and the application verifies that it belongs to the same user. Old versions remain available for audit, but active retrieval excludes superseded facts. See [Phase 2 design](docs/phase-2-advanced-memory.md).

## Phase 3: retrieval evaluation

Compare lexical, semantic, and hybrid retrieval against annotated query-to-memory relevance sets. Reports include Precision@K, Recall@K, MRR, and per-query retrieved IDs. This is an offline retrieval benchmark; semantic/hybrid modes use embeddings when available and hybrid falls back to lexical retrieval.

```bash
python -m src.ltm.cli evaluate --user demo --dataset data/retrieval_eval.json --mode lexical --k 5
python -m src.ltm.cli evaluate --user demo --dataset data/retrieval_eval.json --mode hybrid --k 5 --output results/hybrid.json
```

See [Phase 3 evaluation guide](docs/phase-3-evaluation.md). GitHub Actions runs the unit test suite on pushes and pull requests. LongMemEval is a next integration target; the current harness does not claim full benchmark compatibility.

## Phase 4: LongMemEval benchmark

Download the official benchmark (kept local and git-ignored):

```bash
python scripts/download_longmemeval.py --variant s
```

Run session-level retrieval metrics:

```bash
python -m src.ltm.cli longmemeval --dataset data/longmemeval_s_cleaned.json --k 5 --limit 50
```

Optionally generate QA hypotheses with Gemini (one generation request per question):

```bash
python -m src.ltm.cli longmemeval --dataset data/longmemeval_s_cleaned.json --k 5 --limit 10 --generate-answers
```

The adapter reports Precision@K, Recall@K, MRR, Hit@1, and per-question-type results. No-gold-evidence/abstention questions are counted separately and excluded from retrieval averages. Use the official LongMemEval GPT-4o judge script to score generated hypotheses; see [Phase 4 guide](docs/phase-4-longmemeval.md). Retrieval metrics and end-to-end answer accuracy must be reported separately.


## Phase 5: memory-aware multi-format content creation

Create blogs, articles, social posts, emails, newsletters, reports, scripts, and product descriptions using a structured brief plus relevant user-scoped long-term memories. The feature uses the configured Gemini model; it does not train a new model.

Generate a blog post:

```bash
python -m src.ltm.cli content create --user demo --topic "Why long-term memory matters for AI assistants" --type blog --audience "technology students" --tone "clear and educational" --length medium --keywords memory RAG retrieval --output results/ai-memory-blog.md
```

Generate a report and request a separate editorial review:

```bash
python -m src.ltm.cli content create --user demo --topic "Evaluation of a conversational memory prototype" --type report --audience "research supervisors" --tone formal --length long --review --output results/memory-report.md
```

Revise a draft without overwriting the source:

```bash
python -m src.ltm.cli content revise --input results/ai-memory-blog.md --instructions "Shorten the introduction and add a practical example." --output results/ai-memory-blog-revised.md
```

See [Memory-aware content creation guide](docs/content-creation.md) for architecture, supported formats, design decisions, limitations, and a research evaluation plan. Generated drafts are not automatically stored as factual memories. Human review is required before publication.
