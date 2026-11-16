# CPU TinyLM follow-up results — 17 November 2026

This report records the completed CPU-only GitHub Actions run [#107](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43949507585), on branch `research/long-term-memory-starter`, commit `b3f733a924fbe9f00a8bd1c1959d8c4814b4b0bb`.

- Full tiny-LM workflow: passed all 20 steps.
- General test workflow [#292](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43949503345): passed.
- Artifact: [tiny-memory-lm-wikitext2-pilot-43949507585](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43949507585/artifacts/14266222458), artifact ID `14266222458`.
- The artifact contains byte and BPE checkpoints, training logs, held-out metrics, and familiar/unseen QA outputs.

## Held-out WikiText-2

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Vocabulary size | 256 | 512 |
| Parameters | 445,952 | 478,720 |
| Training steps | 1,000 | 1,000 |
| Held-out cross-entropy (nats per original byte) | 2.4382 | 2.1735 |
| Byte-normalized perplexity | 11.4529 | 8.791 (approx.) |

BPE's held-out cross-entropy normalized by original bytes is about 10.9% lower. BPE also uses fewer tokens per input byte (about 0.627 in the earlier measured tokenizer comparison). This is a language-modeling/context-efficiency result, not proof of better instruction following.

## Familiar paraphrase QA (16 questions)

| Measurement | Byte | BPE |
|---|---:|---:|
| Exact generated answers, no memory | 0/16 | 0/16 |
| Exact generated answers, relevant memory | 0/16 | 2/16 |
| Exact generated answers, curated irrelevant memory | 0/16 | 0/16 |
| Mean token F1, relevant memory | 0.0469 | 0.2747 |
| Mean gold-answer NLL, no memory | 3.5668 | 3.0969 |
| Mean gold-answer NLL, relevant memory | 3.0149 | 2.2382 |
| Mean gold-answer NLL, irrelevant memory | 3.6208 | 3.6867 |
| Relevant minus no-memory NLL | -0.5519 | -0.8587 |
| Irrelevant minus no-memory NLL | +0.0540 | +0.5898 |
| Retrieved evidence contains expected answer | 16/16 | 16/16 |

BPE produced two exact answers on this familiar paraphrase set; byte produced none. Both remain poor generators. Token F1 measures lexical overlap, not semantic correctness. The familiar cases are paraphrases of facts represented in the synthetic instruction-tuning corpus, so they do not establish generalization to new fact values.

## Unseen-fact QA (16 questions over 8 held-out fact values)

| Measurement | Byte | BPE |
|---|---:|---:|
| Exact generated answers, no memory | 0/16 | 0/16 |
| Exact generated answers, relevant memory | 0/16 | 0/16 |
| Exact generated answers, curated irrelevant memory | 0/16 | 0/16 |
| Mean token F1, relevant memory | 0.0313 | 0.0000 |
| Mean gold-answer NLL, no memory | 4.9473 | 5.7391 |
| Mean gold-answer NLL, relevant memory | 4.8309 | 5.7526 |
| Mean gold-answer NLL, irrelevant memory | 4.9957 | 5.6384 |
| Relevant minus no-memory NLL | -0.1164 | +0.0136 |
| Irrelevant minus no-memory NLL | +0.0484 | -0.1007 |
| Retrieved evidence contains expected answer | 16/16 | 16/16 |

The relevant-memory prompt slightly lowered byte-model mean gold-answer NLL, but slightly raised BPE mean NLL. Neither model generated an exact answer in any unseen-fact condition. The eight fact clusters are a small synthetic sample; these aggregate values are descriptive, not strong statistical evidence.

## Interpretation

1. **Retrieval works on this controlled set:** lexical retrieval exposed answer-containing evidence in all 16 unseen cases for both models.
2. **Generation remains the bottleneck:** the retrieval-only evidence check is not a model-generated answer. Both models scored 0/16 exact on unseen facts.
3. **Familiar-set gains do not transfer:** BPE's 2/16 familiar exact answers did not reproduce on unseen fact values.
4. **No evidence of weight-based long-term learning:** memory remains external SQLite retrieval context inserted into the prompt. This experiment does not update weights with user memories or demonstrate continual learning.
5. **The model is not a useful assistant yet:** smoke-test generation remains incoherent, and the small synthetic QA set is not a general benchmark.

The run includes the newline answer-boundary stopping correction. It is an inference-protocol change; it does not change the model weights. Do not interpret this single run as an isolated causal estimate of the stopping change without a controlled same-checkpoint before/after comparison.

## Reproducibility and constraints

The models were randomly initialized and trained on WikiText-2 using CPU GitHub Actions. No pretrained weights, hosted generation API, API key, or paid compute were used. The evaluation data is synthetic, small, and author-created. The result supports retaining BPE as the more efficient tokenizer for further experiments, but not scaling model size or compute yet. Next work should improve answer grounding/decoding and expand fact-clustered unseen evaluation before making capability claims.
