# Joint-span width-penalty sweep (development only)

**Status:** completed on CPU; no training or optimizer updates were performed.

## Protocol

- Checkpoints: the three joint-span seed checkpoints (13, 29, 47) from workflow run 45007297492.
- Development set: `examples/tiny_lm_memory_qa_dev_facts.jsonl` — 16 cases across 8 held-out fact clusters.
- Penalties: 0, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0 score units per predicted byte.
- Each seed was evaluated on all 16 cases; zero cases were skipped.
- Selection: highest mean exact-span accuracy across seeds; ties broken by lower mean joint-span rank, then lower penalty.
- The 64-case unseen-facts final-test fixture was not used for this sweep.

## Results

| Penalty | Mean exact-span accuracy | Mean joint-span rank |
|---:|---:|---:|
| 0.0 | **37.50%** | **111.27** |
| 0.1 | 37.50% | 113.02 |
| 0.2 | 37.50% | 132.52 |
| 0.4 | 37.50% | 171.54 |
| 0.6 | 35.42% | 207.65 |
| 0.8 | 16.67% | 236.21 |
| 1.0 | 8.33% | 254.40 |

Per-seed exact-span accuracy at penalties 0.0 through 0.4 was 37.5% for all three seeds. At 0.6 it was 37.5%, 37.5%, and 31.25%; at 0.8 it was 12.5%, 25%, and 12.5%; at 1.0 it was 6.25%, 6.25%, and 12.5% (seeds 13, 29, 47 respectively).

## Decision

The selection rule chooses **0.0 penalty**. Positive width penalties did not improve exact-span accuracy; among the tied top-accuracy settings, the unpenalized decoder also had the best mean rank. Increasing the penalty generally worsened ranking and eventually harmed exact accuracy.

This is a negative result: the current simple per-byte width penalty is not supported by this development experiment. Keep the decoder unpenalized; do not claim a gain from length regularization.

## Limitations

The development set is small (8 independent facts, with two paraphrased questions per fact) and synthetic. Results are a tuning signal, not a robust estimate of performance on natural conversations. The checkpoints were fine-tuned on the same training recipe but with three random seeds; this is a post-hoc decoder sweep, not independent pretraining replication.

The final unseen-facts test set was not inspected or used for selection. The selected setting is the existing baseline (0.0), whose final-test evaluation is already reported in the joint-span seed replication results; no additional test-set tuning or rerun was performed here.
