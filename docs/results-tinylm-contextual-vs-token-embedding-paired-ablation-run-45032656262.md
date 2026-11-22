# Paired contextual vs token-embedding memory-key ablation

**Workflow:** [Run 45032656262](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45032656262)  
**Evaluated code commit:** `956c7b127dae6a26ee60d3b125eaa09dc2d12638`  
**Artifact:** [tiny-memory-lm-wikitext2-pilot-45032656262](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45032656262/artifacts/14758333134) (expires 2026-12-22)

## Protocol

Paired per-seed runs compared contextual hidden-state keys with raw token-embedding keys. Each pair used the same copy-enabled base checkpoint, seed, training examples, optimizer settings, training steps, and evaluation cases. Both the language model and span head were fine-tuned; this is not a frozen-backbone head-only comparison.

- Seeds: 13, 29, 47
- Training: 600 steps, batch size 4, learning rate 1e-4
- Evaluation: 57 eligible unseen-fact questions per run
- Candidate protocol: all ordered spans with start <= end over retained memory bytes; no gold answer length or length-conditioned candidate windows
- 7 of 64 cases were skipped because the answer span was missing or outside retained context

## Results

| Seed | Contextual exact | Token-embedding exact | Paired delta | Contextual mean rank | Token-embedding mean rank |
|---|---:|---:|---:|---:|---:|
| 13 | 42.11% (24/57) | 31.58% (18/57) | +10.53 pp | 28.04 | 66.40 |
| 29 | 52.63% (30/57) | 31.58% (18/57) | +21.05 pp | 36.28 | 64.95 |
| 47 | 49.12% (28/57) | 28.07% (16/57) | +21.05 pp | 26.40 | 58.23 |
| **Mean** | **47.95%** | **30.41%** | **+17.54 pp** | **30.24** | **63.19** |

Contextual keys won on exact-span accuracy in all three paired seeds (3 wins, 0 ties, 0 losses). Mean rank was lower for contextual keys in all three pairs.

## Interpretation and limitations

This paired experiment supports the hypothesis that position-aware contextual representations materially improve this joint span selector relative to raw token embeddings in this setup. It strengthens the earlier observation because representation mode is the only intended experimental difference within each seed pair.

The result is still preliminary: only three seeds, a small narrow extractive evaluation set, and seven skipped cases. It measures exact extraction of answer spans from retained memory, not free-form answer generation, broad QA quality, durable memory over long conversations, or performance on diverse real-world memories. Since both the backbone and span head are fine-tuned in both conditions, it does not isolate a frozen-backbone representation effect.

## Reproducibility

The workflow artifact contains the paired summary JSON, per-seed/per-mode JSON metrics, logs, and checkpoints under `joint-span-key-ablation/`. The checked-in runner is `scripts/run_tiny_lm_joint_span_key_ablation.py`; the mode switch is `--memory-key-mode contextual|token_embedding` in `scripts/train_tiny_lm_joint_span.py`.
