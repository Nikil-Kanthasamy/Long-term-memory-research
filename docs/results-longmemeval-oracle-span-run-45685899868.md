# LongMemEval-S Oracle-Evidence Span Experiment

- Workflow: [45685899868](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45685899868)
- Branch: `research/long-term-memory-starter`
- Evaluated commit: `5a0b9d9c0bec6bafc6fbe810f1a5ac1fcca5bef4`
- Workflow conclusion: success
- Dataset: official LongMemEval-S cleaned JSON, 500 questions
- Protocol: oracle-evidence extractive span localization; not retrieval and not end-to-end question answering

## Dataset construction and coverage

The builder retained only questions for which a `has_answer=true` evidence turn contained the exact reference answer string after deterministic truncation to a 128-byte context block. It retained 20 of 500 questions (4.0%) and skipped 480.

| Eligibility outcome | Questions |
|---|---:|
| Eligible | 20 |
| Abstention | 30 |
| Answer not literal in retained evidence | 342 |
| Invalid question or answer | 32 |
| Question exceeds context | 76 |
| Total skipped | 480 |

Eligible question types:
- knowledge-update: 4
- single-session-assistant: 1
- single-session-user: 12
- temporal-reasoning: 3

This is a highly filtered subset, not a representative estimate across all LongMemEval-S questions. In particular, 342 questions were excluded because the reference answer was not a literal substring in the retained evidence; this protocol therefore favors extractive answers and does not measure semantic answer reconstruction.

## Model and training protocol

The three runs used seeds 13, 29, and 47. Each started from the same copy-enabled base checkpoint, then fine-tuned the language model and joint-span head for 600 steps. The span selector was not given the gold answer length. The model was supplied a gold evidence turn, so evidence retrieval was bypassed.

## Results

| Seed | Examples | Exact joint-span accuracy | Start top-1 | End top-1 | Mean joint-span rank |
|---:|---:|---:|---:|---:|---:|
| 13 | 20 | 0.0% | 15.0% | 25.0% | 564.00 |
| 29 | 20 | 0.0% | 15.0% | 20.0% | 493.95 |
| 47 | 20 | 10.0% | 5.0% | 15.0% | 510.65 |
| **Mean** | **20 per seed** | **3.33%** | **11.67%** | **20.00%** | **522.87** |

Only one seed localized an exact answer span, and it did so on 2 of 20 examples. Exact accuracy is averaged across three runs on the same 20 examples; the 60 run-example measurements are not 60 independent questions.

## Interpretation

1. **Transfer is poor.** The contextual joint-span approach that performed better on small synthetic fixtures does not transfer to this filtered LongMemEval subset.
2. **The evaluation is favorable to the model in one important respect.** It receives a gold evidence turn. A complete system would also need to retrieve the right turn among a user's history.
3. **The eligible subset is very small and biased.** The 20 examples are selected by exact string matching and 128-byte truncation. Results must not be generalized to the full 500-question benchmark.
4. **This does not establish end-to-end LongMemEval performance.** No retrieval score, generated-answer score, semantic match score, abstention score, or temporal-reasoning score was measured here.
5. **The current result is a useful negative result.** It shows that success on synthetic span-localization tests is insufficient evidence of robust conversational-memory answering.

## Artifact issue and correction

The workflow completed all LongMemEval steps, but its upload-artifact path list omitted:
- `longmemeval-span-eval.jsonl`
- `longmemeval-span-eval-build-summary.json`
- `longmemeval-oracle-span-seed-sweep/`
- `longmemeval-oracle-span-seed-sweep.log`

The result summaries were therefore visible in the workflow logs but not included in the downloadable artifact. The workflow has been corrected on the research branch to include these outputs in subsequent runs. This report records the verified values from the completed workflow log.

## Next experiments

1. Preserve the full LongMemEval subset and per-seed outputs in the next artifact; validate the output files and expected row counts before upload.
2. Add a non-extractive/semantic evaluation path so reference answers need not appear verbatim in a truncated evidence turn.
3. Expand coverage using a clearly specified evidence-window policy, including multi-turn evidence, without silently changing the evaluation denominator.
4. Build a separate retrieval evaluation over LongMemEval sessions, then evaluate retrieval and answer generation jointly.
5. Include abstention and temporal-update cases explicitly rather than excluding them from the main evaluation.
6. Keep a held-out split and report per-question-type results with confidence intervals once the sample size supports them.

## Overall conclusion

The LongMemEval-S experiment is complete and the workflow passed, but the measured result is weak: 3.33% mean exact span accuracy on only 20 highly filtered oracle-evidence questions. The next milestone should not be a larger claim about system performance. It should be a better evaluation protocol that covers semantic answers, retrieval, multi-turn evidence, temporal updates, and abstention, with complete artifacts and transparent denominators.
