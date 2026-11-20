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


---

## Follow-up experiment: expanded 288-example grounding curriculum

- Source commit: `438ed567c85126604c1a7cab102800cf80a5293c`
- Full CPU workflow: [run #115](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44663920889)
- General unit tests: [run #317](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44663920908)
- Both workflows completed successfully.
- Artifact: `tiny-memory-lm-wikitext2-pilot-44663920889`
- The generated curriculum contains 288 examples (16 facts × 3 memory phrasings × 6 question phrasings). The SFT checkpoint metadata reports **336 total training examples**, including the 48 examples in the base demo set.
- Both models used 2,000 SFT steps. The byte model's base checkpoint was step 900; BPE's was step 1,000.

### Held-out unseen-fact results after the expanded curriculum

| Model | No-memory exact | Relevant-memory exact | Irrelevant-memory exact | Relevant-memory mean token F1 | Retrieval evidence contains answer |
|---|---:|---:|---:|---:|---:|
| Byte | 0/64 | 0/64 | 0/64 | 0.0313 | 59/64 (92.2%) |
| BPE | 0/64 | 0/64 | 0/64 | 0.0383 | 59/64 (92.2%) |

The byte model's relevant-memory token-F1 gain over no memory was 0.0313, with a cluster-bootstrap 95% interval of [0.0000, 0.0875]. BPE's gain was 0.0260, with interval [-0.0045, 0.0662]. Neither model improved exact-match accuracy: every one of the 32 fact clusters had zero exact-match delta for both models.

### Teacher-forced gold-answer likelihood

| Model | No-memory mean NLL | Relevant-memory mean NLL | Relevant minus no-memory | Cluster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Byte | 4.5190 | 4.1772 | -0.3419 | [-0.5293, -0.1700] |
| BPE | 5.4597 | 5.2516 | -0.2081 | [-0.3681, -0.0415] |

The likelihood improvement is statistically separated from zero under this exploratory cluster bootstrap for both tokenizers. It still does not translate into correct generated answers. This distinction is central: the model assigns somewhat higher probability to the correct continuation when relevant evidence is present, but its greedy generation does not reproduce the answer.

### Familiar-set regression check

| Model | Relevant-memory exact | Relevant-memory mean token F1 |
|---|---:|---:|
| Byte | 0/16 | 0.0000 |
| BPE | 0/16 | 0.1047 |

Retrieval evidence contained the expected answer in all 16 cases for both tokenizers. The earlier BPE familiar-set exact answers did not reproduce in this follow-up run.

### Conclusion and next experiment

The larger and more varied synthetic curriculum improved teacher-forced answer likelihood on the unseen benchmark, but did **not** improve exact generated-answer accuracy. The model is still not a reliable memory QA assistant. The evidence points to an answer-copying/decoding bottleneck rather than simply too few prompt phrasings.

The next experiment should test an explicit copy mechanism or pointer-style output distribution over retrieved context, with a controlled copy benchmark and untouched held-out facts. Keep the current benchmark as a regression test; do not claim weight-based persistent memory from retrieval-context results. No paid compute or external generation API was used.


---

## Follow-up experiment: causal pointer-copy head

- Source commit: `049284d5618ae0292dbb68cd31b64d6d5d251e8f`
- Full CPU workflow: [run #116](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/44666049942)
- General tests and every step of the full model workflow completed successfully.
- Artifact: `tiny-memory-lm-wikitext2-pilot-44666049942`
- The copy-enabled byte model was randomly initialized and trained on the same WikiText-2 pipeline (1,000 pretraining steps), then instruction-tuned for 2,000 steps on the 336-example grounding set. It has 462,465 parameters.
- The optional causal pointer-copy head combines the vocabulary distribution with a learned attention distribution over tokens already in the input context. The baseline byte and BPE models remain unchanged.

### Results

| Evaluation | No-memory exact | Relevant-memory exact | Irrelevant-memory exact | Relevant-memory mean token F1 | Retrieval evidence contains answer |
|---|---:|---:|---:|---:|---:|
| Familiar (16 questions) | 0/16 | 0/16 | 0/16 | 0.0000 | 16/16 (100%) |
| Unseen (64 questions) | 0/64 | 0/64 | 0/64 | 0.0266 | 59/64 (92.2%) |

The unseen relevant-memory token-F1 delta over no memory was +0.0266 (cluster-bootstrap 95% interval [0.0000, 0.0594]). The corresponding baseline byte model from the preceding run had +0.0313 relevant-memory token F1. Neither model produced a single exact answer on the unseen set; the copy head did not improve free-generation accuracy.

### Teacher-forced answer likelihood

| Evaluation | No-memory mean NLL | Relevant-memory mean NLL | Relevant minus no-memory | Cluster-bootstrap 95% interval |
|---|---:|---:|---:|---:|
| Familiar (16 questions) | 3.2065 | 2.4586 | -0.7478 | [-1.1353, -0.4282] |
| Unseen (64 questions) | 4.3609 | 4.0048 | -0.3562 | [-0.5693, -0.1688] |

Relevant retrieved context improves teacher-forced gold-answer likelihood on both synthetic sets. However, the unseen NLL improvement is only marginally different from the baseline byte model's -0.3419 delta, while its mean token F1 is slightly lower (0.0266 versus 0.0313). Irrelevant memory also worsened unseen mean NLL relative to no memory (4.4953 versus 4.3609). These results do not show a useful generation improvement.

### Conclusion

The pointer-copy head trains, passes unit tests, and changes answer-token likelihood, but it does **not** solve the generation bottleneck. Exact generated-answer accuracy remains 0/64 on unseen facts and 0/16 on familiar facts. The current copy distribution operates over input tokens and is not evidence of learned persistent memory in model weights; SQLite retrieval still supplies external context. Do not present this model as a reliable factual assistant.

A better next step is to isolate copying from language modeling with a small, explicitly supervised span-copy task: train the model to select the answer span from a retrieved passage, evaluate exact span reproduction on held-out fact values, and separately measure distractor resistance. Keep the existing QA set untouched as the end-to-end regression benchmark. No paid compute or hosted generation API was used.
