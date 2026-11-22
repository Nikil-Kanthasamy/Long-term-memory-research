# Contextual-memory-key joint-span replication

**Workflow:** [Run 45028977280](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45028977280)  
**Commit evaluated:** `1a50cdfe3493f25fffc8fcaa8bab33e16df49f32`  
**Branch:** `research/long-term-memory-starter`  
**Runtime:** GitHub Actions, CPU-only  
**Artifact:** [tiny-memory-lm-wikitext2-pilot-45028977280](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45028977280/artifacts/14757476429) (26,881,278 bytes; expires 2026-12-22)

## Result

The full workflow completed successfully. The corrected joint-span model uses contextual, position-aware Transformer hidden states as memory keys rather than raw byte-token embeddings. The model and span head were both fine-tuned.

On the 57 eligible unseen-fact cases (7 of 64 skipped because the answer span was not present in the retained, answer-independent memory prefix), three-seed exact-span accuracy was:

| Seed | Exact span accuracy | Exact matches | Start top-1 | End top-1 | Mean joint rank |
|---:|---:|---:|---:|---:|---:|
| 13 | 42.11% | 24/57 | 70.18% | 57.89% | 28.04 |
| 29 | 52.63% | 30/57 | 52.63% | 66.67% | 36.25 |
| 47 | 49.12% | 28/57 | 49.12% | 42.11% | 26.39 |
| **Mean** | **47.95%** | **27.33/57** | — | — | — |

Across seeds, exact-span accuracy ranged from 42.11% to 52.63% (10.53 percentage-point spread). The single joint-span run in this workflow achieved 49.12% (28/57), with mean joint rank 26.39. Before training, that run had 0% exact-span accuracy and mean rank 832.44.

The previous raw-embedding-key replication averaged 30.41% exact-span accuracy (17.33/57) across three seeds. The contextual-key replication is 17.54 percentage points higher on the same 57-case evaluation protocol. This is an encouraging controlled follow-up, but it is still a small pilot with only three seeds and one narrow dataset; it does not establish generalization to other memory distributions or free-form answer generation.

## Other span results in the same workflow

- Gold-length-conditioned span-localization diagnostic: gold span mass increased from 0.0615 to 0.8010 and top-span accuracy reached 52.63%. This supplies gold answer length when forming candidates, so it is **not** answer-length-free extraction.
- Independent answer-length-free start/end classifier: exact span accuracy 5.26% (3/57).
- Joint answer-length-free span selector: exact span accuracy 49.12% (28/57).

The 64-question test set was used for evaluation only; 288 extractive training examples were prepared, while 48 paraphrased/non-verbatim answers were skipped. Seven evaluation cases were omitted because the answer was outside the retained context or absent. The three-seed runner used the same copy-enabled base checkpoint initialization for each seed, independently initialized span heads, and fine-tuned both the language model and span head for 600 steps. No gold answer length was supplied to joint-span candidate scoring.

## Interpretation and next checks

The result supports the hypothesis that contextual, position-aware memory representations are more useful span keys than repeated raw byte embeddings. It does not isolate representation choice from every possible optimization effect, and it does not prove the raw-key representation was the sole cause of prior errors. Next work should add a paired ablation that holds initialization, training order, and seed fixed while varying only the key representation, then evaluate on a larger independently constructed memory-QA set and report confidence intervals.

All workflow stages, including tests, training, evaluation, three-seed replication, and artifact upload, completed successfully. No changes were made to `main`; the work remains on the research branch and PR #1 remains unmerged.
