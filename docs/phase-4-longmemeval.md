# Phase 4: LongMemEval integration

## What this adds

The adapter ingests each benchmark question's timestamped history as session-level memory records in a temporary SQLite database. Each question uses its own isolated user namespace, so records from different questions cannot leak across runs. It supports two session-level retrieval modes:

- **Lexical Jaccard**: the original token-overlap baseline with importance weighting.
- **BM25**: Okapi term-frequency/inverse-document-frequency scoring with document-length normalization; the importance bonus is applied only to positive matches.

It scores ranked session IDs against `answer_session_ids` and reports Precision@K, Recall@K, MRR, Hit@1, and per-question-type breakdowns. Questions with no gold evidence sessions (the benchmark's 30 abstention cases) are excluded from retrieval averages and counted separately, consistent with the official benchmark retrieval protocol.

## Download the official data

The downloader fetches the cleaned Hugging Face release. Dataset files are stored under `data/`, which is git-ignored; do not commit benchmark data.

```bash
python scripts/download_longmemeval.py --variant s
# Optional: oracle-only history or the much larger M variant
python scripts/download_longmemeval.py --variant oracle
```

The S dataset is the recommended first run. The M dataset has substantially larger histories and memory requirements.

## Retrieval-only evaluation

Lexical baseline:

```bash
python -m src.ltm.cli longmemeval \
  --dataset data/longmemeval_s_cleaned.json \
  --retrieval-mode lexical --k 5 \
  --output results/longmemeval_lexical.json
```

BM25:

```bash
python -m src.ltm.cli longmemeval \
  --dataset data/longmemeval_s_cleaned.json \
  --retrieval-mode bm25 --k 5 \
  --output results/longmemeval_bm25.json
```

Add `--limit 50` for a quick pilot; omit it for all 500 questions. Each report includes per-question retrieved session IDs and scores.

## Recorded retrieval comparison

The paired 500-question S-split run completed successfully on GitHub Actions:

- [Workflow run 43500776110](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43500776110)
- [Detailed results and limitations](longmemeval-s-retrieval-results.md)

BM25 improved Recall@5 to 91.44% and Hit@1 to 87.02%, versus 28.73% and 11.91% for lexical Jaccard. This is retrieval-only performance; it is not an end-to-end QA score. The results report records per-category scores and caveats.

## Retrieval-augmented answer generation

The CLI can use the configured Gemini reader to generate a hypothesis from the top-K retrieved sessions. For example:

```bash
python -m src.ltm.cli longmemeval \
  --dataset data/longmemeval_s_cleaned.json \
  --retrieval-mode bm25 --k 5 \
  --limit 10 --generate-answers \
  --predictions results/longmemeval_hypotheses.jsonl
```

Set `GEMINI_API_KEY` in `.env` first. Each question triggers one generation request, so start with a small limit and check current API quotas/pricing. The JSONL output uses the official benchmark's `question_id` and `hypothesis` fields. The generation code asks the reader to use only retrieved evidence, treat conversation text as evidence rather than instructions, and abstain when evidence is insufficient.

## Official answer judging

The LongMemEval authors provide an official QA evaluator. Download the matching `longmemeval_oracle.json` and use the [official repository](https://github.com/xiaowu0162/LongMemEval). Install its lightweight evaluation requirements, then from its `src/evaluation` directory run:

```bash
export OPENAI_API_KEY=YOUR_API_KEY
python evaluate_qa.py gpt-4o \
  /path/to/longmemeval_hypotheses.jsonl \
  /path/to/longmemeval_oracle.json
```

The official judge uses a separate OpenAI API key; it is not the Gemini answer-generation key. It writes an evaluation log with per-question labels and reports aggregate QA accuracy. Keep the hypotheses, oracle dataset version, model identifiers, prompts, and code revision together.

## Scope and limitations

- The benchmark adapter currently retrieves whole sessions. It does not yet implement turn-level retrieval, dense embeddings, hybrid reranking, key expansion, or time-aware query pruning.
- The BM25 result is one deterministic run at K=5; audit per-question errors and reproduce before making strong comparative claims.
- Optional QA generation uses Gemini and is not itself the official judge. Retrieval scores and answer accuracy must be reported separately.
- The full S dataset is roughly hundreds of MB. The M dataset is much larger; begin with S.
- No benchmark data, generated hypotheses, API keys, or personal conversation data should be committed.

## Run from GitHub Actions

A manual workflow is available at **Actions → LongMemEval retrieval benchmark → Run workflow** on the `research/long-term-memory-starter` branch. The workflow downloads the cleaned S dataset, runs lexical and BM25 against the same split and K, and uploads both JSON reports as a 30-day artifact. It requires no LLM API key because it is retrieval-only.

## Next milestone: end-to-end memory assistant

The repository already has a Gemini client, an optional retrieval-augmented reader, memory extraction/update primitives, SQLite persistence, and a retrieval CLI. The next major milestone is a full end-to-end evaluation: run all 500 questions through BM25 retrieval plus Gemini answer generation, then grade the predictions with the official GPT-4o judge. That requires a Gemini API key and an OpenAI API key (or another approved judge setup). The estimated engineering effort is 2–4 focused days for a first reproducible end-to-end result, then 1–2 weeks for hybrid retrieval, error analysis, and reliability improvements. API execution time and usage charges are additional and depend on the selected models and quotas.
