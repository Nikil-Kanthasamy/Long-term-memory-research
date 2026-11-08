# Long-Term Memory Research

A Python research sandbox for persistent memory in language-model applications. It runs alongside a separate ML training pipeline and does not modify model weights.

## Baseline
- One content-generation model: Google Gemini API (gemini-2.5-flash by default) for generation, memory extraction, and consolidation.
- SQLite persistence for conversation turns and extracted memory records.
- Optional Gemini embeddings (gemini-embedding-001) with cosine similarity; keyword fallback works without an API key.
- Memory lifecycle: extract, retrieve, deduplicate, supersede (planned), and delete.
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
