# TinyLM expanded memory experiment — 20 November 2026

## Reproducibility

- Repository branch: `research/long-term-memory-starter`
- Source commit: `3166e19b27d065a25a582a94e42502dab84ecffc`
- Workflow run: [#311](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44662525634)
- Artifact: `tiny-memory-lm-wikitext2-pilot-44662525634`
- Execution device: CPU GitHub Actions
- Both the general test workflow and every step of the full TinyLM workflow completed successfully.
- Models were randomly initialized and trained locally in the workflow; no hosted text-generation API or API key was used.

## Experiment design

The workflow trained a small decoder-only Transformer on WikiText-2, then supervised-instruction-tuned it using the repository's synthetic instruction examples. It compared a UTF-8 byte tokenizer and a custom byte-level BPE tokenizer.

The expanded unseen-fact benchmark contains 64 questions over 32 independently authored synthetic fact clusters (two paraphrases per cluster). Each case is evaluated with no memory, relevant retrieved memory, and curated irrelevant memory. The report also records whether the retrieval evidence itself contains the expected answer.

Cluster bootstrap intervals use 2,000 seeded resamples of fact clusters, not individual paraphrase rows.

## Held-out language-model results

| Model | Tokenizer | Training checkpoint step | Test CE (nats / original byte) | Perplexity |
|---|---|---:|---:|---:|
| Byte | UTF-8 byte | 900 | 2.4382 | 11.45 per byte-token |
| BPE | byte-level BPE | 1,000 | 2.1735 | 31.92 per BPE token |

The BPE model's byte-normalized cross-entropy was lower on this test split. Per-token perplexities are not directly comparable across tokenizers; the byte-normalized cross-entropy is the more useful comparison here.

## Unseen-fact memory QA (64 questions, 32 fact clusters)

### Retrieval and free generation

| Model | No-memory exact | Relevant-memory exact | Irrelevant-memory exact | Retrieval evidence contains answer |
|---|---:|---:|---:|---:|
| Byte | 0/64 | 0/64 | 0/64 | 59/64 (92.2%) |
| BPE | 0/64 | 0/64 | 0/64 | 59/64 (92.2%) |

Mean generated token F1 for the byte model was 0.0000 without memory, 0.0045 with relevant memory, and 0.0000 with irrelevant memory. For BPE it was 0.0078, 0.0245, and 0.0035 respectively.

**Interpretation:** retrieval is substantially ahead of generation. The memory layer usually exposes the expected fact, but neither small model reliably converts that evidence into a correct free-form answer. Zero exact matches means this prototype is not yet a useful factual QA assistant.

### Teacher-forced likelihood

Teacher-forced gold-answer negative log-likelihood (NLL) measures how probable the expected answer tokens are when supplied as the target. Lower is better, but this is not free-generation accuracy or proof of reasoning.

| Model | No-memory mean NLL | Relevant-memory mean NLL | Irrelevant-memory mean NLL | Relevant minus no-memory |
|---|---:|---:|---:|---:|
| Byte | 4.8841 | 4.7961 | 5.0073 | -0.0880 |
| BPE | 5.2364 | 5.0605 | 5.4914 | -0.1759 |

Cluster-bootstrap 95% interval for the byte relevant-memory NLL delta: **[-0.2546, 0.0982]**. This interval includes zero.

Cluster-bootstrap 95% interval for the BPE relevant-memory NLL delta: **[-0.3292, -0.0353]**. This interval is below zero on this synthetic benchmark.

For BPE, the relevant-memory token-F1 delta was +0.0167, with a 95% interval of [-0.0146, 0.0479], which includes zero. Exact-match accuracy did not improve for either model.

Irrelevant memory increased mean NLL relative to no memory for both models (+0.1232 byte, +0.2550 BPE), which is consistent with distractor context sometimes hurting prediction.

## Familiar-set comparison

The earlier 16-question familiar paraphrase set produced the following results:

- Byte model: 0/16 exact generated answers with relevant memory; mean token F1 0.0469.
- BPE model: 2/16 exact generated answers with relevant memory; mean token F1 0.2747.
- Retrieval evidence contained the answer in 16/16 cases for both models.

The small familiar-set BPE result does not generalize to the expanded unseen-fact set: BPE scored 0/64 exact answers on unseen facts.

## What this establishes — and what it does not

1. The from-scratch training and evaluation pipeline runs end-to-end on CPU.
2. The retrieval component can expose answer-bearing evidence for most cases in this controlled synthetic benchmark.
3. Relevant memory changes the small model's teacher-forced answer-token likelihood; for BPE, the cluster-bootstrap interval was below zero in this benchmark.
4. The model still fails at reliable free-form factual answering: 0/64 exact unseen answers for both tokenizers.
5. The experiment does **not** demonstrate that the model learned persistent facts into its weights. Retrieved SQLite memories are external context inserted into the prompt. This is retrieval-augmented inference, not continual weight learning.
6. The benchmark uses manually authored synthetic facts and literal answer matching. It is not a substitute for evaluation on real user memories or human-judged semantic answers.

## Next research priorities

- Improve answer extraction/copying from retrieved evidence before increasing model size or compute.
- Add a controlled copy-and-grounding task with a held-out fact split, while keeping the current unseen-fact set untouched for regression testing.
- Measure retrieval quality separately from generation quality and include distractor robustness.
- Investigate training objectives that teach the model to condition on retrieved passages and reproduce exact entities/numbers.
- Only claim progress when free-generation exact match and semantic answer quality improve on held-out facts; teacher-forced likelihood alone is insufficient.

## Artifact contents

The workflow artifact includes byte and BPE checkpoints (best, final, and instruction-tuned), training logs, held-out language-model metrics, familiar and unseen memory-QA reports, and a memory-chat smoke-test output.
