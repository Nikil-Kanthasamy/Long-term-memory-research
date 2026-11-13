# Phase 2: Advanced memory engine

## Implemented
- Additive SQLite schema migration for valid_from, valid_until, and supersedes_id.
- Versioned updates: an explicitly identified active memory can be marked superseded while its historical record remains.
- User-scope validation prevents a memory update from superseding another user's record.
- Time-aware retrieval excludes facts outside their validity window.
- Raw conversation retrieval complements extracted facts, reducing detail loss from summarization.
- CLI commands: history and timeline.

## Write-path behavior
The extractor receives a bounded list of active memories with IDs. It may attach supersedes_id only when a new statement clearly corrects an existing fact. The application validates the ID against the same user's active memories before updating. If extraction or the API fails, the original turn remains stored. This is not a guarantee against model mistakes: evaluate update precision before trusting automatic writes on important data.

## Temporal semantics
valid_from and valid_until are ISO-8601 timestamps. Missing valid_until means no known expiry. A superseded memory remains in the database for audit/history but is excluded from active retrieval. For historical queries, pass an explicit as_of timestamp to the store search API.

## Research caution
Raw-turn retrieval currently uses lexical overlap, while fact retrieval supports optional dense embeddings. This is a reproducible baseline, not a production-grade hybrid ranker. Next evaluate reciprocal-rank fusion, query expansion, temporal query parsing, and reranking on LongMemEval. The original benchmark evaluates information extraction, multi-session reasoning, temporal reasoning, knowledge updates, and abstention (ICLR 2025): https://proceedings.iclr.cc/paper_files/paper/2025/hash/d813d324dbf0598bbdc9c8e79740ed01-Abstract-Conference.html
