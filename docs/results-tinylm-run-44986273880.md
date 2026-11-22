# TinyLM pointer-copy and answer-span results — 22 November 2026

## Reproducibility

- GitHub Actions run: [44986273880](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44986273880)
- Source branch: `research/long-term-memory-starter`
- Source commit: `ecd746c2c66e6eeb0332188a212848824df80820`
- Runner: GitHub-hosted CPU runner; no paid accelerator or hosted generation API.
- Artifact: `tiny-memory-lm-wikitext2-pilot-44986273880` (17,210,490 bytes; retained until 22 December 2026).
- All substantive workflow steps completed successfully, including byte/BPE training, pointer-copy tuning, QA evaluation, and span supervision.

The model weights were randomly initialized and trained in this workflow. The separate SQLite memory store supplies retrieved text in the model context; retrieval is not itself a weight update or continual-learning mechanism.

## Language-model baseline

| Model | Test cross-entropy | Perplexity / normalization |
|---|---:|---:|
| Byte tokenizer | 2.4382 nats/token | 11.45 per byte-token |
| BPE tokenizer | 3.4631 nats/BPE token | 2.1735 nats/byte (approx.) |

The byte and BPE token-level losses are not directly comparable. BPE uses fewer tokens; byte-normalized cross-entropy is the more useful comparison here. These small models are trained for a limited number of CPU steps and are not general-purpose assistants.

## Synthetic memory QA

The unseen-fact benchmark contains 64 questions over 32 fact clusters. Retrieval placed the literal expected answer in its evidence for 59/64 questions (92.2%) for the evaluated variants. This is a retrieval/evidence-exposure result, not a generated-answer score.

| Model | Exact generated answers with relevant memory | Mean answer token F1 with relevant memory | Relevant-memory minus no-memory gold-answer NLL |
|---|---:|---:|---:|
| Byte | 0/64 | 0.0313 | -0.3419 |
| BPE | 0/64 | 0.0383 | -0.2081 |
| Pointer-copy | 0/64 | 0.0266 | -0.3562 |

The pointer-copy model's cluster-bootstrap 95% interval for the relevant-memory minus no-memory NLL delta was [-0.5693, -0.1688] across 32 fact clusters. The BPE interval was [-0.3681, -0.0415]. The byte interval was [-0.5293, -0.1700].

On the 16-question familiar benchmark, the pointer-copy model also produced 0/16 exact answers with memory. Its mean teacher-forced gold-answer NLL decreased from 3.2065 without memory to 2.4586 with relevant memory (delta -0.7478). This likelihood change must not be described as correct answer generation.

Across the 64 unseen questions, irrelevant-memory pointer-copy NLL was 0.1343 nats worse than no-memory NLL, while relevant-memory NLL was 0.3562 nats better. This suggests the conditioning signal is not simply a universal benefit from adding arbitrary context, but the generated-answer results remain poor.

## Answer-span supervision

The span experiment starts from the pointer-copy checkpoint and trains a span-selection head on synthetic memory-grounding examples.

| Diagnostic | Before span supervision | After span supervision |
|---|---:|---:|
| Scorable held-out cases | 57 | 57 |
| Mean probability mass on gold span | 0.0615 | 0.8010 |
| Top-span accuracy | 0.0% | 52.6% |
| Mean gold-span rank | 21.81 | 3.58 |
| Mean candidate spans | 39.35 | 39.35 |

The training set contained 288 examples. Forty-eight paraphrased answers were skipped because the expected answer was not a verbatim span in the memory text. On evaluation, seven cases were skipped because the answer span was absent from the retained context prefix; seven cases are therefore not included in the 57-case span metrics.

**Important protocol limitation:** context truncation is answer-independent deterministic head truncation, avoiding the earlier answer-conditioned/oracle crop. However, the span-ranking diagnostic supplies the gold answer's byte length to define candidate windows. It is therefore a constrained localization test, not answer-length-free extraction or end-to-end QA. The top-span accuracy is based on exact span match.

## What this does and does not establish

1. The retrieval layer can expose relevant fact text on most of this small synthetic unseen-fact benchmark.
2. Relevant memory measurably changes teacher-forced likelihood of gold answer tokens for the byte, BPE, and pointer-copy models.
3. Span supervision teaches the model's selection head to concentrate probability mass on annotated answer spans under the stated constrained protocol.
4. **Free-form answer generation remains unsuccessful:** exact answer accuracy is zero on the 64-question unseen benchmark for all three model variants.
5. This is not evidence that writing a fact to SQLite changes the language model's weights. The span-head training does update weights, but it learns a synthetic selection behavior; it does not demonstrate that newly stored facts are continually learned into the model.
6. The benchmark is synthetic, small, and manually authored. It does not establish performance on real user histories, long conversations, multi-hop reasoning, or robust semantic retrieval.

## Next research steps

- Remove the gold-answer-length input from span candidate generation and train/evaluate a joint start/end or boundary-prediction objective.
- Evaluate on longer, independently authored memory passages with controlled distractors and answer positions, including facts near the end of the context window.
- Improve answer decoding and measure exact/semantic answer quality separately from teacher-forced likelihood.
- Add tests that ensure all reported denominators and skipped cases are explicit.
- Investigate actual continual-learning mechanisms separately (e.g. controlled parameter updates or adapters) and compare them against retrieval-only memory with forgetting and interference measurements.

## Artifact contents

The run artifact includes byte, BPE, and pointer-copy checkpoints; training and instruction-tuning logs; familiar and unseen QA JSON; span-supervision logs and evaluation JSON; and held-out WikiText-2 metrics. It is available from the linked GitHub Actions run under **Artifacts**.
