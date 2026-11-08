# Research plan

## Question
Can compact, curated external memory improve factual recall and personalized content generation across sessions without sending full conversation history to the language model?

## Hypotheses
- H1: hybrid retrieval improves recall over keyword-only retrieval.
- H2: extracted memory uses fewer prompt tokens than raw-history stuffing at similar accuracy.
- H3: temporal/conflict handling reduces stale-fact errors.
- H4: write quality (precision, duplication, update correctness) matters as much as retrieval.

## Baselines
A. No memory: current request only.
B. Full history: all prior turns, token-budget controlled.
C. Raw retrieval: retrieve matching history chunks.
D. Extracted memory: retrieve atomic facts.
E. Hybrid: facts plus selected source snippets.

## Procedure
1. Create a fixed dataset of synthetic and consented multi-session conversations.
2. Annotate facts, preferences, events, timestamps, updates, and answerable questions.
3. Split by user/conversation, not random message, to prevent leakage.
4. Run A-E with identical answer model, prompt budget, and decoding settings.
5. Evaluate extraction independently from end-to-end question answering.
6. Record model IDs, prompts, timestamps, quota failures, and code revision.

## Metrics
- Retrieval Precision@k and Recall@k.
- Extraction precision, recall, and F1 against annotated durable facts.
- Answer accuracy; human rubric for open-ended generation.
- Temporal and contradiction/update accuracy.
- Unsupported-claim rate.
- Duplicate and stale-memory rates.
- Retrieved tokens, latency, API calls, and estimated cost.

## Ablations
Semantic vs lexical vs hybrid; top-k and threshold; extraction prompt; confidence threshold; importance weighting; raw vs summarized memory; consolidation frequency; local vs hosted embeddings.

## Training integration
Keep memory separate from the training loop. Export curated, anonymized, consented examples only. Compare inference-time retrieval, fine-tuning, and both. Maintain a held-out user/session split to detect leakage.

## Experiment log
| Run | Commit | Dataset | Write model | Embedding | Retrieval | Top-k | Accuracy | Recall@k | Tokens | Latency | Notes |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| baseline-001 | TBD | synthetic-v1 | gemini-2.5-flash | gemini-embedding-001 | hybrid | 5 | TBD | TBD | TBD | TBD | Initial implementation; not benchmarked |

## Acceptance criteria
Reproducible versioned dataset; no cross-user leakage; deletion verified; report accuracy with token/latency/cost; comparable context budgets across baselines.
