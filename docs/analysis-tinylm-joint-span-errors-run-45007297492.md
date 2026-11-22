# Joint-span error analysis — workflow run 45007297492

**Date:** 2026-11-22  
**Source:** the three per-seed evaluation JSON files in artifact [14748244103](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45007297492/artifacts/14748244103)  
**Base evaluation:** 57 extractive cases per seed, 29 distinct fact clusters (28 paired paraphrase variants and one singleton).

## Main finding

The earlier interpretation that *end-boundary prediction is the main bottleneck* was incorrect. It treated the independent end-logit argmax as if it were the end selected by the joint span decoder.

Reconstructing gold byte offsets from each retained memory and comparing them with the actual joint-decoder outputs gives:

| Joint-decoder boundary metric | Seed 13 | Seed 29 | Seed 47 | Mean across seeds |
|---|---:|---:|---:|---:|
| Start boundary exact | 45.61% | 45.61% | 42.11% | **44.44%** |
| End boundary exact | 63.16% | 66.67% | 63.16% | **64.33%** |
| Both boundaries exact | 31.58% | 31.58% | 28.07% | **30.41%** |
| Mean absolute start offset (bytes) | 8.46 | 8.49 | 8.46 | **8.47** |
| Mean absolute end offset (bytes) | 3.07 | 2.16 | 3.04 | **2.75** |

By contrast, the separate marginal top-1 diagnostics in the original JSON average 38.60% for start and 15.20% for end. Those are not the boundary accuracies of the joint-decoder's selected pair. The compatibility term changes the selected pair, substantially improving end localization.

## Error pattern by gold answer byte length

Pooled across the three seeds (171 predictions; repeated paraphrase variants remain correlated):

| Gold answer bytes | Predictions | Exact spans | Joint start exact | Joint end exact | Mean absolute start error | Mean absolute end error |
|---|---:|---:|---:|---:|---:|---:|
| 1–4 | 24 | 0.00% | 25.00% | 25.00% | 24.33 | 1.75 |
| 5–10 | 96 | 35.42% | 47.92% | 70.83% | 6.38 | 2.13 |
| 11+ | 51 | 35.29% | 47.06% | 70.59% | 4.94 | 4.41 |
| **All** | **171** | **30.41%** | **44.44%** | **64.33%** | **8.47** | **2.75** |

The most visible failure mode is a **start boundary placed too early**, creating an overlong span whose end can nevertheless be close to the correct end. All 24 pooled predictions for 1–4-byte answers failed exact span matching. Several recurring examples select the correct or near-correct end but start 29–35 bytes too early. These short-answer results come from four fact clusters, so they are a diagnostic signal rather than a broad statistical estimate.

## Revised research direction

1. Do not prioritize the independent end head based only on its 15.2% marginal top-1 score; that metric misstates joint-decoder behavior.
2. Focus on start-boundary localization and overlong-span errors, especially for short answers.
3. Compare candidate-span score calibration against a fixed, answer-independent length prior or width penalty. Tune it only on a development split, then report the untouched evaluation separately.
4. Add per-case gold/predicted offsets, predicted width, and boundary errors to future evaluation JSON so these diagnostics are generated automatically.

## Limitations

- The same 57 cases are evaluated for each seed. There are 29 distinct fact clusters; many are represented by two paraphrase variants, so the 171 seed-case predictions are not independent observations.
- The bucket counts are pooled across seeds and must not be interpreted as 171 independent test examples.
- The analysis uses the actual joint-decoder predicted offsets and gold strings located in the corresponding retained memories. It does not measure free-form generation or retrieval quality.
- No new model was trained and no evaluation split was used for hyperparameter selection in this analysis.


## Reproducible diagnostics

The offline analyzer now also reports signed start offsets (negative means predicted start is too early), early/late start fractions, overlong/short span fractions, and the pooled per-case rows. This makes it possible to audit the direction of the error rather than relying only on absolute offsets.

Run it from the repository root after downloading and extracting the workflow artifact:

```bash
python scripts/analyze_tiny_lm_joint_span_errors.py \
  --cases examples/tiny_lm_memory_qa_unseen_facts.jsonl \
  --seed-results joint-span-seed-sweep/joint-span-seed-13.json \
                 joint-span-seed-sweep/joint-span-seed-29.json \
                 joint-span-seed-sweep/joint-span-seed-47.json \
  --block-size 128 \
  --output joint-span-error-analysis.json
python -m unittest discover -s tests -p 'test_tiny_lm_joint_span_error_analysis.py' -v
```

The analyzer and its tests are post-processing utilities; they do not retrain the model or tune any parameter against the evaluation set. The newly added signed-error test cases were checked against the helper logic, but a repository CI run has not yet been triggered for these post-processing-only files.
