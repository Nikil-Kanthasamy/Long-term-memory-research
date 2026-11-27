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

- Session-level and turn-level lexical/BM25 retrieval are implemented. Dense embeddings, hybrid reranking, query expansion, and explicit time-aware ranking remain future work.
- The BM25 result is one deterministic run at K=5; audit per-question errors and reproduce before making strong comparative claims.
- Optional QA generation uses Gemini and is not itself the official judge. Retrieval scores and answer accuracy must be reported separately.
- The full S dataset is roughly hundreds of MB. The M dataset is much larger; begin with S.
- No benchmark data, generated hypotheses, API keys, or personal conversation data should be committed.

## Run from GitHub Actions

A manual workflow is available at **Actions → LongMemEval retrieval benchmark → Run workflow** on the `research/long-term-memory-starter` branch. The workflow downloads the cleaned S dataset, runs lexical and BM25 against the same split and K, and uploads both JSON reports as a 30-day artifact. It requires no LLM API key because it is retrieval-only.

## Next milestone: end-to-end memory assistant

The repository already has a Gemini client, an optional retrieval-augmented reader, memory extraction/update primitives, SQLite persistence, and a retrieval CLI. The next major milestone is a full end-to-end evaluation: run all 500 questions through BM25 retrieval plus Gemini answer generation, then grade the predictions with the official GPT-4o judge. That requires a Gemini API key and an OpenAI API key (or another approved judge setup). The estimated engineering effort is 2–4 focused days for a first reproducible end-to-end result, then 1–2 weeks for hybrid retrieval, error analysis, and reliability improvements. API execution time and usage charges are additional and depend on the selected models and quotas.


## Turn-level evidence retrieval (new benchmark)

The full-session benchmark can find the right conversation while still returning too much irrelevant text to a reader. The companion `scripts/build_longmemeval_turn_retrieval_eval.py` benchmark ranks individual turns and uses only official `has_answer=true` turn labels as relevance judgments. It reports Precision@K, Recall@K, Hit@K, MRR, per-question-type results, answerable-question coverage, abstentions, and skipped cases. It compares transparent token-set Jaccard and BM25 under the same question-level haystack.

Run it after downloading LongMemEval-S:

```bash
python scripts/build_longmemeval_turn_retrieval_eval.py \
  --input data/longmemeval_s_cleaned.json \
  --output-cases results/longmemeval_s_turn_cases.jsonl \
  --output-summary results/longmemeval_s_turn_retrieval.json \
  --k 10
```

This is a retrieval-only diagnostic, not answer generation or semantic correctness. Gold relevance is the benchmark's turn-level `has_answer` annotation. Questions without any labeled evidence turn are counted as coverage failures rather than silently assigned a zero-relevance ranking; abstention questions are reported separately. The automated workflow now runs this full-S turn benchmark and uploads the per-question cases with the summary. In [workflow run 45925646437](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45925646437), the index contained 231,595 turns; all 470 answerable questions were eligible, with 30 abstentions reported separately. BM25 achieved Recall@5 67.15%, Hit@5 79.36%, and MRR 66.30%; Jaccard achieved 41.72%, 52.55%, and 40.01%. The per-question-type breakdown shows particularly low BM25 Recall@5 for multi-session (47.36%) and single-session-preference (33.33%) questions. See [the complete results and limitations](results-longmemeval-turn-retrieval-run-45925646437.md). These figures measure exact annotated evidence-turn retrieval only, not answer correctness. The same run's session-level BM25 Recall@5 was 91.44%; that number is not directly comparable because the unit and relevance labels differ. The turn-level gap points to evidence localization and query matching as the next retrieval bottlenecks, before testing reader synthesis.
