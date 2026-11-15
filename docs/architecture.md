# Architecture and design decisions

1. **Ingestion:** append each input turn to SQLite before network calls. Raw turns provide provenance; they are not automatically injected into prompts.
2. **Extraction:** one Gemini text model proposes standalone durable facts as JSON. Extraction failure must not lose the raw turn.
3. **Memory record:** content, category, importance, confidence, source turn, timestamps, status, and optional embedding.
4. **Deduplication:** exact duplicate guard in this baseline. Next add semantic duplicate detection and conflict resolution.
5. **Retrieval:** cosine similarity over stored Gemini embeddings plus lexical overlap. Without an API key, lexical retrieval is used.
6. **Generation:** retrieved memory is supplied as evidence with an explicit abstention instruction.
7. **Deletion:** user-scoped hard delete by memory ID.

## Memory classes
- Working: current request/context window; transient.
- Episodic: dated interaction or event, with source provenance.
- Semantic: stable facts, preferences, entities, and relationships.
- Procedural: reusable workflow or instruction.
- Raw history: source material for future re-extraction, subject to retention policy.

## Next iterations
- Add valid_from, valid_until, and supersedes_id for changing facts.
- Add local embeddings only after measuring CPU/GPU impact on model training.
- Add consolidation to merge duplicates and mark contradictions as superseded without erasing source history.
- Add access control, encryption at rest, retention controls, and export/import.
- Ask for confirmation before persisting uncertain personal details.

## Privacy
Treat retrieved memory as untrusted data, not system instructions. Scope every query by user ID. Avoid storing secrets or sensitive personal data. API calls transmit supplied text to Google; review free-tier data-use terms before real conversations. Keep .env and data/ out of version control.
