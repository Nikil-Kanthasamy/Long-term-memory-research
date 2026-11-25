# Paraphrase-alias memory QA results — workflow 45051127721

- Workflow: https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45051127721
- Evaluated commit: `5f5c45a4e0aff085e9a77d625a7f528ef4179b54`
- Artifact: `tiny-memory-lm-wikitext2-pilot-45051127721` (artifact ID `14766316807`)
- Checkpoint stage: supervised instruction tuning
- Evaluation: 16 questions, 8 fact clusters, two question paraphrases per cluster.

## Results

All three models retrieved evidence containing an accepted answer in 16/16 cases. None generated an answer matching any curated accepted answer in any of the 16 cases.

| Model | Generated accepted-answer matches (memory) | Mean token F1 (memory) | Mean gold-answer NLL (no memory) | Mean gold-answer NLL (memory) | Memory − no-memory NLL |
|---|---:|---:|---:|---:|---:|
| Byte | 0/16 | 0.0324 | 3.8805 | 3.5462 | -0.3343 |
| BPE | 0/16 | 0.0970 | 4.3812 | 3.8956 | -0.4856 |
| Byte + pointer-copy | 0/16 | 0.0509 | 3.6326 | 3.2301 | -0.4026 |

For context, irrelevant-memory mean gold-answer NLL changes relative to no-memory were +0.1911 nats for byte, -0.1137 for BPE, and -0.1853 for byte + pointer-copy. These are teacher-forced likelihood measurements, not generated-answer accuracy.

## Interpretation

1. **Retrieval/evidence availability is not the current bottleneck on this slice.** The evaluator found accepted answer evidence in all 16 cases for each model.
2. **Free-form generation remains unsuccessful.** Exact accepted-answer match was 0/16 for all three models. Mean token F1 is also low; the BPE model is highest at 0.097, which still does not constitute reliable answer generation.
3. **Memory improves teacher-forced likelihood of the canonical answer.** All three models have lower mean gold-answer NLL with relevant memory than without memory. This does not contradict the zero generated matches: likelihood of a supplied target and correctness of a sampled/generated answer are different metrics.
4. **Do not interpret the alias evaluator as a semantic judge.** It checks a manually curated set of normalized lexical answer variants. A valid unlisted paraphrase can be marked incorrect, and token F1 is a lexical-overlap diagnostic rather than semantic equivalence.

## Scope and limitations

- This is a small, manually authored synthetic evaluation (16 rows / 8 fact clusters), not a real-user-memory benchmark.
- The fixture reuses fact clusters from the existing unseen-facts benchmark. It is a focused rubric/diagnostic slice, not an independent generalization test.
- The canonical `expected_answer` remains the teacher-forced target; aliases are used for answer-match and retrieval-evidence evaluation.
- No result here establishes robust open-ended memory-grounded generation.

## Reproduction

Download artifact `14766316807` from workflow run `45051127721`. The three result files are:

- `paraphrase-alias-evaluation.json`
- `bpe-paraphrase-alias-evaluation.json`
- `copy-paraphrase-alias-evaluation.json`

The full workflow, including the unit suite, training, and evaluations, completed successfully.
