# TinyLM pointer-copy pilot results — workflow 44901506551

**Branch:** `research/long-term-memory-starter`  
**Commit tested:** `8199049b276e23a85d03bcc76c1bd314722fc828`  
**Workflow:** [GitHub Actions run 44901506551](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44901506551)  
**Execution:** CPU-only GitHub Actions; no hosted generation API or API key.

## Outcome

The complete workflow passed, including unit tests, randomly initialized language-model training, held-out WikiText-2 evaluation, memory QA, pointer-copy training, and the answer-span supervision stage.

The pointer-copy model learned to place substantially more causal copy-attention mass on a gold answer span in retrieved memory after 600 supervised updates:

| Metric on scorable unseen QA rows | Before span supervision | After span supervision |
|---|---:|---:|
| Rows scored | 60 | 60 |
| Mean attention mass on gold span | 0.0661 | 0.8007 |
| Gold span ranked first | 0.0% | 50.0% |
| Mean gold-span rank (lower is better) | 22.17 | 4.07 |
| Mean candidate windows per row | 39.25 | 39.25 |

The span trainer prepared 288 extractive training examples from 16 synthetic training facts. It skipped 48 examples whose answer was not a verbatim substring of the memory. The held-out evaluation fixture contains 64 questions over 32 separate synthetic fact clusters; 60 rows had extractable answer spans and 4 were skipped for the same reason.

The training facts and held-out QA facts are separately authored. The held-out questions use two paraphrases per fact cluster.

## Important evaluation caveats

This is a **span-localization diagnostic**, not a general extraction or answer-generation result:

- The evaluator supplies the expected answer to find and preserve its span when fitting a long memory passage into the model's 128-byte context. This is oracle, answer-conditioned cropping; a deployed retriever cannot use the expected answer.
- Candidate windows are defined using the gold answer's byte length. The model is not asked to predict answer boundaries or answer length.
- The 4 non-extractive held-out rows are excluded from span metrics. Reported span metrics therefore describe the 60 scorable rows, not all 64.
- The evaluation is synthetic and small. It does not establish performance on natural conversations or real user memories.
- The 50% top-span result is meaningful progress over the 0% baseline, but still means half of scorable examples do not rank the correct window first.

## Generation and retrieval remain separate bottlenecks

On the same pointer-copy checkpoint before the span-supervision stage:

| Unseen-fact QA metric | Result |
|---|---:|
| Exact generated answers with relevant memory | 0 / 64 |
| Mean generated-answer token F1 with relevant memory | 0.0266 |
| Retrieved evidence contains expected answer | 59 / 64 (92.2%) |
| Relevant-memory minus no-memory teacher-forced answer NLL | -0.3562 |

The teacher-forced NLL improvement means the expected answer receives higher likelihood under relevant-memory conditioning. It is **not** generated-answer accuracy. The 600-step span-supervision stage is evaluated for attention localization; this report does not claim it fixed free-form generation.

## What is learning, and what is memory?

The base language model is randomly initialized and trained locally on WikiText-2. During the span-supervision stage, gradient updates change model parameters so its causal copy-attention mechanism more often points toward a supplied answer span in the prompt.

The factual content at inference time is still supplied as retrieved memory text in the prompt. This experiment does **not** demonstrate that user facts are written into model weights through continual learning. It demonstrates a limited learned behavior for selecting a span from externally supplied context.

## Reproduction

The workflow's span stage runs:

```bash
python scripts/train_tiny_lm_span_copy.py \
  --input /tmp/tiny-lm-memory-grounding-sft.jsonl \
  --checkpoint models/tiny-memory-lm-copy/best.pt \
  --output-checkpoint models/tiny-memory-lm-copy/span-copy.pt \
  --eval-cases examples/tiny_lm_memory_qa_unseen_facts.jsonl \
  --output copy-span-unseen-evaluation.json \
  --steps 600 --batch-size 4 --device cpu
```

The full workflow artifact is attached to [run 44901506551](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44901506551) and retained by GitHub Actions for 30 days.

## Next experiment

The next useful test is to remove both oracle conveniences: use a realistic, answer-independent truncation/retrieval policy and predict span boundaries without supplying gold answer length. Evaluate at the fact-cluster level and report confidence intervals. Only after that should this be treated as a candidate extractive memory component. A separate generation objective and exact-answer evaluation are still required before calling the system a useful memory-augmented assistant.
