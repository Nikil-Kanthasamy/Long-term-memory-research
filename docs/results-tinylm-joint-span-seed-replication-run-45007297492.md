# Joint-span seed replication — workflow run 45007297492

**Date:** 2026-11-22  
**Branch:** `research/long-term-memory-starter`  
**Commit under test:** `30f51095d2c413b4ce09b9a827c9c96a498708a0`  
**Workflow:** [tiny LM model tests, run #132](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45007297492)  
**Artifact:** [tiny-memory-lm-wikitext2-pilot-45007297492](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45007297492/artifacts/14748244103) (26,883,856 bytes; retained until 2026-12-22)

## Result

The answer-length-free joint-span selector reproduced the prior 28.07% exact-span result across three independent joint-head initialization seeds. Mean exact-span accuracy was **30.41%** (an average of 17.33 exact matches per 57-case evaluation), with a range of **28.07–31.58%**.

| Seed | Evaluation cases | Start top-1 | End top-1 | Exact joint span | Mean joint-span rank |
|---:|---:|---:|---:|---:|---:|
| 13 | 57 | 38.60% | 14.04% | 31.58% (18/57) | 66.33 |
| 29 | 57 | 38.60% | 15.79% | 31.58% (18/57) | 64.98 |
| 47 | 57 | 38.60% | 15.79% | 28.07% (16/57) | 58.30 |
| **Mean** | **57 per seed** | **38.60%** | **15.20%** | **30.41%** | **63.20** |

- Absolute range: 3.51 percentage points.
- The original single-run joint-span result was 28.07% (16/57); seed 47 matched it exactly.
- All runs initialized from the same copy-enabled base checkpoint, then jointly optimized the language-model parameters and independently initialized joint-span head for each seed (600 steps, batch size 4, CPU, same evaluation cases).
- Candidate scoring considers all ordered spans with start <= end over retained memory bytes. **Gold answer length is not supplied** to define candidate windows.

## Context and limitations

- Evaluation size is 57 eligible cases per seed. Seven cases were skipped because the answer span was missing or outside the retained answer-independent context.
- The training data prepared 288 joint-span examples; 48 non-extractive/paraphrased rows were skipped, and zero rows were skipped for exceeding context.
- The 30.41% mean is a small-set extractive span-selection metric, not free-form answer accuracy, not a retrieval metric, and not evidence of broad generalization.
- This sweep varies joint-head initialization and independently fine-tunes the language model from the same starting checkpoint for each seed. It is therefore a multi-seed fine-tuning replication, not three independent from-scratch pretraining runs.
- Earlier gold-length-conditioned span-mass/top-span diagnostics are not directly comparable to this answer-length-free evaluation.

## Workflow status

Workflow #132 completed successfully. Unit tests and all preceding experiment stages passed; the three-seed sweep and artifact upload completed. The uploaded artifact contains the checkpoints, evaluation JSON, and training logs for inspection.

## Reproduction command

```bash
python scripts/run_tiny_lm_joint_span_seeds.py \
  --input /tmp/tiny-lm-memory-grounding-sft.jsonl \
  --checkpoint models/tiny-memory-lm-copy/best.pt \
  --eval-cases examples/tiny_lm_memory_qa_unseen_facts.jsonl \
  --output-dir joint-span-seed-sweep \
  --seeds 13 29 47 \
  --steps 600 \
  --batch-size 4 \
  --device cpu
```
