# TinyLM joint-span extraction results — 2026-11-22

## Run provenance

- GitHub Actions run: [#130](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45006507442)
- Branch: `research/long-term-memory-starter`
- Commit: `0ca54c3a96ca426d687fd565095811c9db665c97`
- Workflow conclusion: success; the unit-test workflow and full CPU training/evaluation job both passed.
- Artifact: [tiny-memory-lm-wikitext2-pilot-45006507442](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45006507442/artifacts/14748500564) (21,006,661 bytes; retained by GitHub for 30 days from upload).

## Answer-length-free span results

All three diagnostics use the same 57 unseen-fact evaluation cases with an answer span present in the retained, answer-independent memory context. Seven cases had no usable exact answer span or fell outside the retained prefix and were skipped.

| Metric | Independent boundary head | Joint span head |
|---|---:|---:|
| Start top-1 accuracy | 45.61% | 38.60% |
| End top-1 accuracy | 7.02% | 12.28% |
| Exact start-and-end span accuracy | 3.51% | **28.07%** |
| Mean gold candidate rank | start 3.60; end 3.81 | 58.21 |

The joint head scores all ordered spans with start <= end over retained memory bytes. It does not receive the gold answer length and does not use length-conditioned candidate windows. It improved exact-span accuracy from 2/57 to 16/57 relative to the independent-boundary head, despite lower start-only top-1 accuracy.

The independent boundary head's valid predicted-span rate was 19.30%; this metric is not directly equivalent to joint exact-span accuracy and should not be compared as if it were.

## Training data and diagnostics

- 288 training examples were prepared for each of the boundary and joint-span heads.
- 48 non-extractive/paraphrased answers were skipped because the target answer was not a verbatim span in memory.
- No training examples exceeded the context limit.
- Evaluation used 57 cases; 7 were skipped because the exact answer span was missing or outside the answer-independent retained context.
- Joint-span training ran for 600 steps on CPU. The logged loss was 6.8079 at step 1 and 2.6174 at step 600; the trajectory was non-monotonic.

## Interpretation and limitations

This is a promising result for selecting complete memory spans: joint scoring substantially outperformed independently selecting start and end positions on this pilot. It is not evidence of robust open-ended answer generation. The evaluation set is small, contains only extractive cases with a verbatim answer in the retained context, and excludes paraphrased/non-extractive targets. The result should be replicated on a larger held-out set with multiple seeds and compared against simple lexical/heuristic extractors before claiming a general memory-QA improvement.

The earlier span-localization experiment that reported 52.63% top-span accuracy supplied the gold answer length to define candidate windows; it is a different, easier diagnostic and is not comparable to the answer-length-free exact-span metric reported here.

## Next experiments

1. Repeat the joint-span experiment with at least three random seeds and report mean and variance.
2. Expand the evaluation set and stratify extractive, paraphrased, multi-fact, temporal, and unanswerable questions.
3. Compare joint-span extraction with BM25/lexical matching and a simple longest-overlap baseline.
4. Measure whether extracted spans improve answer exact match/F1 when passed to a separate answer renderer; keep extraction and generation metrics separate.
5. Preserve the current artifact and checkpoint as the reproducible baseline.

No changes were made to `main`; PR #1 remains open and unmerged.
