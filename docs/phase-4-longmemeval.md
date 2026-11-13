# Phase 4: LongMemEval integration

## What this adds

The adapter ingests each benchmark question's timestamped history as session-level memory records in a temporary SQLite database. Each question uses its own isolated user namespace, so records from different questions cannot leak across runs. It retrieves the top-K sessions using the project's lexical Jaccard baseline and scores against `answer_session_ids`.

Reported retrieval metrics:
- Precision@K
- Recall@K
- Mean Reciprocal Rank (MRR)
- Hit@1
- Per-question-type breakdown

Questions with no gold evidence sessions (the benchmark's abstention cases) are excluded from retrieval averages and counted separately. This is consistent with the benchmark authors' retrieval evaluation note that abstention cases have no ground-truth answer location.

## Download the official data

The downloader fetches the official Hugging Face release. Dataset files are stored under `data/`, which is git-ignored; do not commit benchmark data.

```bash
python scripts/download_longmemeval.py --variant s
# Optional: oracle-only history or the much larger M variant
python scripts/download_longmemeval.py --variant oracle
```

The S dataset is the recommended first run. The M dataset has substantially larger histories and memory requirements.

## Retrieval-only evaluation

```bash
python -m src.ltm.cli longmemeval \
  --dataset data/longmemeval_s_cleaned.json \
  --k 5 \
  --limit 50 \
  --output results/longmemeval_retrieval.json
```

Remove `--limit 50` to run all questions. The report includes each question's retrieved session IDs and scores. This is a reproducible lexical retrieval baseline, not a BM25 or dense-embedding result.

## Retrieval-augmented answer generation

To generate hypotheses using the configured Gemini reader:

```bash
python -m src.ltm.cli longmemeval \
  --dataset data/longmemeval_s_cleaned.json \
  --k 5 \
  --limit 10 \
  --generate-answers \
  --predictions results/longmemeval_hypotheses.jsonl
```

Set `GEMINI_API_KEY` in `.env` first. Each question triggers a generation request, so start with a small limit and check current API quotas/pricing. The output JSONL uses the official benchmark's `question_id` and `hypothesis` fields.

## Official answer judging

The LongMemEval authors provide an answer evaluation script. Clone their [official repository](https://github.com/xiaowu0162/LongMemEval), install its lightweight evaluation requirements, and download the matching oracle dataset. Then, from its `src/evaluation` directory, run:

```bash
python evaluate_qa.py gpt-4o \
  /path/to/longmemeval_hypotheses.jsonl \
  /path/to/longmemeval_oracle.json
```

Configure `OPENAI_API_KEY` for the official GPT-4o judge. This is separate from the Gemini answer-generation key. The judge produces an auto-evaluation log and aggregate QA scores. Keep the hypothesis file, dataset variant, model IDs, and code revision together when recording results.

## Scope and limitations

- The adapter benchmarks **session-level retrieval**. It does not yet implement the paper's BM25/dense retrievers, turn-level retrieval, key expansion, or time-aware query pruning.
- The current retriever is the repository's token-set Jaccard lexical scorer with importance weighting. Do not label it BM25.
- Optional QA generation uses Gemini and is not itself the official judge. Use the authors' evaluator for comparable answer correctness.
- Retrieval scores are not end-to-end QA scores. Report them separately.
- The full S dataset is roughly hundreds of MB. The M dataset is much larger; begin with S and a small `--limit`.
- No benchmark data, generated hypotheses, or personal conversation data should be committed.

## Run from GitHub Actions (no local dataset download required)

A manual workflow is available at **Actions → LongMemEval retrieval benchmark → Run workflow** on the `research/long-term-memory-starter` branch.

- The default `limit=50` runs a small retrieval pilot. Set the limit to a larger number, or leave it blank, to evaluate the full S split.
- `k=5` is the default retrieval depth.
- The workflow downloads the cleaned S dataset at run time, runs retrieval only (no Gemini API key required), and uploads `longmemeval_s_retrieval.json` as a 30-day artifact.
- The dataset and result are not committed to the repository. The workflow is manual-only and does not run on every push.

Use the artifact to inspect aggregate metrics and per-question retrieval results. Record the workflow run URL, commit SHA, limit, K, and artifact together when reporting results.
