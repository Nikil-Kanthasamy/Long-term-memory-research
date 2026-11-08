# Phase 3: retrieval evaluation

## Goal

Measure retrieval independently from answer generation. This phase provides a deterministic, offline harness for annotated query-to-memory relevance sets. It avoids model calls unless semantic or hybrid retrieval is configured with an embedding function.

## Dataset format

Use a JSON array or JSONL file. Each case identifies a query and the IDs of relevant memory records already loaded into the same SQLite database:

```json
[
  {
    "query": "Which framework is used for vision training?",
    "relevant_memory_ids": [12, 18]
  }
]
```

Optional `user_id` on a case overrides the CLI's default user. Keep test users isolated and never mix private user memories into public benchmark data.

## Run

```bash
python -m src.ltm.cli evaluate --user demo --dataset data/retrieval_eval.json --mode lexical --k 5
python -m src.ltm.cli evaluate --user demo --dataset data/retrieval_eval.json --mode hybrid --k 5 --output results/hybrid.json
python -m unittest discover -s tests -v
```

Modes:
- `lexical`: token overlap baseline.
- `semantic`: embedding cosine similarity (requires working embedding API).
- `hybrid`: weighted semantic + lexical score; falls back to lexical when embeddings are unavailable.

Metrics are macro-averaged Precision@K, Recall@K, and reciprocal rank (MRR). Per-query retrieved IDs and scores are saved in the JSON report. Use identical data, user scope, K, and relevance annotations when comparing runs.

## LongMemEval integration

LongMemEval contains 500 questions across six labels and tests single-session recall, multi-session reasoning, temporal reasoning, knowledge updates, and preferences; abstention is represented in its question set. See the [official repository](https://github.com/xiaowu0162/LongMemEval) and [ICLR 2025 paper](https://proceedings.iclr.cc/paper_files/paper/2025/file/d813d324dbf0598bbdc9c8e79740ed01-Paper-Conference.pdf).

The harness here is a retrieval-metric foundation, **not a complete LongMemEval runner**. LongMemEval gold evidence is expressed as source session IDs, whereas this harness scores memory-record IDs. A future adapter must map session evidence to ingested memory records and separately implement the benchmark's answer-generation and judging protocol. Do not compare retrieval Recall@K with end-to-end QA accuracy.

## Limitations

- The current lexical score is token-set Jaccard overlap, not BM25.
- Semantic retrieval quality depends on the selected embedding model.
- No reranker, query expansion, session chunking, judge model, cost accounting, or statistical significance analysis is included yet.
- Results are only meaningful when relevance labels are reviewed and the dataset split prevents user/session leakage.
