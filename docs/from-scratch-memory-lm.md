# From-scratch language model with long-term memory

## Goal

Build a small, inspectable decoder-only Transformer whose weights are initialized randomly and trained locally, then connect it to the existing SQLite memory store. This is a research prototype, not a claim that a small CPU-trained model can match a frontier assistant.

## What is genuinely from scratch

- No pretrained model weights are loaded.
- No hosted generation API is called by the tiny-model training or inference scripts.
- The original baseline uses a fixed UTF-8 byte vocabulary (256 IDs). An experimental byte-level BPE tokenizer is now trained from the training split only; neither path uses a downloaded pretrained tokenizer.
- The decoder-only Transformer, embeddings, causal attention stack, and output head are initialized randomly and trained with next-byte prediction.
- Persistent memory remains external to model weights. At inference, the existing SQLite store retrieves relevant memory snippets and places them before the user question.

The existing Gemini-based application remains intact as a baseline. The new experiment is an optional separate path.

## Initial model target

| Setting | Initial value |
|---|---:|
| Architecture | Decoder-only Transformer / GPT-style |
| Parameters | Approximately 3.2 million (varies slightly by config) |
| Layers | 4 |
| Attention heads | 4 |
| Hidden width | 256 |
| Context window | 256 bytes |
| Vocabulary | 256 UTF-8 byte values |
| Objective | Causal next-token prediction |
| Optimizer | AdamW |
| Initial training | Local CPU or CUDA |
| Persistent memory | Existing user-scoped SQLite store + lexical retrieval |

This model is intentionally small enough for experimentation. Byte-level tokenization is simple and fully reproducible but less compute-efficient than a learned subword tokenizer. The generated text will be weak until trained on a suitable corpus and may remain poor at instruction following.

## Install and train

Python 3.10+ is recommended. Install the optional dependency:

```bash
python -m pip install -r requirements-tiny-lm.txt
```

Use a UTF-8 plain-text corpus that you have permission to use. A reproducible starter corpus is prepared by the included WikiText-2 downloader. For meaningful general language ability, a much larger and more diverse corpus will be needed. Keep held-out validation and test splits separate. Do not use private conversation memories as training data by default.

```bash
python -m pip install -r requirements-tiny-lm-data.txt
python scripts/prepare_wikitext.py --output-dir data/wikitext-2
python scripts/train_tiny_lm.py \
  --input data/wikitext-2/train.txt \
  --validation-input data/wikitext-2/validation.txt \
  --output-dir models/tiny-memory-lm \
  --steps 3000 --device auto
```

The starter corpus is the public WikiText-2 raw split from Salesforce/Hugging Face. Its dataset card describes it as Wikipedia Good/Featured article text and lists CC BY-SA 4.0; retain attribution and comply with the share-alike terms for any distributed adaptations or outputs where applicable. The separate validation split is preserved rather than randomly mixing validation text into training. The test split is prepared but not used for model selection. [Dataset card](https://huggingface.co/datasets/Salesforce/wikitext).

The script prints parameter count, device, corpus hashes, train loss, and validation loss. It saves `best.pt` and `final.pt`. Evaluate the untouched test split with:

```bash
python scripts/evaluate_tiny_lm.py \
  --checkpoint models/tiny-memory-lm/best.pt \
  --input data/wikitext-2/test.txt
```

The evaluator reports next-byte cross-entropy and byte-level perplexity. These are language-modeling metrics, not conversational QA accuracy. Model checkpoints and corpora should remain local and must not be committed to Git.

## Supervised instruction tuning

The base model is only a next-byte predictor. The optional second stage trains on JSONL prompt/response examples and masks the prompt bytes from the loss, so the model is optimized to produce responses rather than copy prompts. The synthetic demonstration set is for pipeline development, not a substitute for a licensed, diverse conversational corpus. The starter demonstration set contains 48 prompt/response examples. A separate deterministic grounding-curriculum generator creates 288 examples from 16 synthetic facts using three memory phrasings and six question phrasings per fact. The generated fact values are checked to remain disjoint from the held-out unseen-fact evaluation set.

```bash
python scripts/train_tiny_lm_sft.py \
  --input examples/tiny_lm_sft_demo.jsonl \
  --checkpoint models/tiny-memory-lm/best.pt \
  --output models/tiny-memory-lm-instruct.pt \
  --steps 100
```

Run the synthetic memory-use comparison against the same checkpoint:

```bash
python scripts/evaluate_tiny_lm_memory_qa.py \
  --checkpoint models/tiny-memory-lm-instruct.pt \
  --cases examples/tiny_lm_memory_qa_eval.jsonl \
  --output results/tiny-lm-memory-qa.json
```

This reports literal normalized answer-substring accuracy for both `no_memory` and `memory` conditions, along with individual outputs and retrieved-memory counts. The held-out evaluation set now contains 16 paraphrased questions across eight facts. It is still a small controlled diagnostic, not a benchmark of general memory reasoning. The CI workflow trains an instruction-tuned checkpoint from the WikiText-2 pilot checkpoint and runs this paired evaluation.

For real experiments, replace the demo with a substantially larger, diverse, reviewed, and appropriately licensed instruction dataset. Keep a held-out prompt/response set and measure exact-answer accuracy, abstention, and memory-vs-no-memory performance. Never treat the demo training loss as evidence of general assistant capability.

## Latest completed CPU pilot (16 November 2026)

Workflow: [run 43749385240](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43749385240). It completed successfully on the research branch. The uploaded artifact contains the checkpoints, logs, held-out metrics, and generated QA outputs.

| Measurement | Result |
|---|---:|
| Parameters | 445,952 |
| Configuration | 2 layers, 4 heads, width 128, context 128 bytes |
| Pretraining steps | 1,000 |
| Final training loss | 2.3898 nats/byte |
| Final validation loss | 2.4075 nats/byte |
| Held-out test perplexity | 11.4529 per byte |
| Instruction-tuning steps | 2,000 |
| SFT response-only loss | 3.0657 to 0.0077 |
| Synthetic QA without memory | 0/6 exact answer matches |
| Synthetic QA with memory | 0/6 exact answer matches |
| Gold answer present in retrieved evidence | 6/6 (100%) |

The memory retriever returned one candidate in each of the six test cases, and the evidence check confirmed that the gold answer text was present in retrieved evidence in all six cases (6/6). Yet neither memory-assisted nor no-memory generation produced an exact expected answer (0/6 in each condition). This isolates the current bottleneck in this tiny controlled set: lexical retrieval found the supplied facts, while the randomly initialized language model could not turn those facts into answers. The chat smoke-test output was also incoherent. Therefore this pilot validates that training, checkpointing, retrieval insertion, and evaluation execute end-to-end; it does **not** validate a useful conversational model or a measurable memory benefit.

The very low instruction-tuning loss alongside zero QA accuracy is a warning that the earlier eight-example demonstration set was being memorized or otherwise overfit, not that the model had learned general instruction following. The next controlled experiment uses the 288-example generated grounding curriculum (16 synthetic facts × three memory phrasings × six question phrasings) alongside the 48-example demonstration set, while evaluating on fact values held out from both sources. This remains synthetic; the result must be judged by held-out free-generation answers, not training loss. The evidence-retrieval score is reported separately from generation quality.

## Chat with retrieved long-term memory

Add an explicitly curated memory using the existing project CLI if the Gemini extraction key is configured, or insert an approved memory directly with the existing `MemoryStore` API. Then run:

```bash
python scripts/chat_tiny_lm.py \
  --checkpoint models/tiny-memory-lm/best.pt \
  --user demo \
  --question "What project am I working on?" \
  --top-k 5
```

The `--remember` option explicitly stores the question text as a conversation memory. It is off by default. The assistant uses lexical retrieval in this first prototype, not the BM25 LongMemEval benchmark path.

## Important limitations

1. A next-byte language model is not automatically a helpful chat assistant. It needs high-quality conversational/instruction training examples and careful evaluation.
2. The memory mechanism is retrieval-augmented context, not weight modification or online learning. This separation makes memory editable, user-scoped, inspectable, and deletable.
3. The first prototype uses lexical retrieval. BM25 integration and dense retrieval can be added after validating the base model. The GitHub Actions tiny-LM workflow runs a short CPU training pilot on WikiText-2, applies a small supervised instruction-tuning stage, evaluates held-out language modeling, compares synthetic memory QA with and without retrieved context, and uploads a 30-day artifact.
4. No corpus is bundled. Dataset selection, licensing, filtering, deduplication, language coverage, and held-out splits must be handled before serious training.
5. CPU training may be slow. A GPU is recommended for larger corpora or longer runs.
6. A tiny model may memorize its training data. Do not train on sensitive user conversations without explicit informed consent, data minimization, and a deletion policy.
7. A real benchmark requires held-out perplexity plus memory QA accuracy, retrieval ablations, hallucination/abstention tests, and comparison with the existing Gemini baseline.

## Development milestones

1. **Model smoke test:** train on a tiny approved corpus; verify loss decreases, checkpoints reload, and generation runs.
2. **Memory smoke test:** add a known memory, retrieve it for a query, and inspect whether the generated continuation uses it.
3. **Instruction tuning:** construct a curated, licensed conversational dataset and train on prompt/response examples with loss masked over prompt tokens.
4. **Memory evaluation:** compare no-memory and retrieved-memory conditions on synthetic and held-out user-specific QA.
5. **Scale responsibly:** evaluate parameter count, context length, data size, GPU memory, wall-clock time, and validation metrics before increasing model size.

## What “full LLM” means here

This creates the neural language-model architecture and training pipeline from scratch, but the first version is a tiny research model, not a competitive general-purpose LLM. A capable model would require a much larger architecture, tokenizer, carefully curated multi-billion-token corpus, substantial accelerator compute, safety tuning, and repeated evaluations. The practical path is to validate this small model and its memory system before scaling.


## Expanded instruction-data experiment (in progress)

The next CPU pilot uses 48 synthetic prompt/response examples (eight facts × six distinct prompt styles) and 16 held-out paraphrase questions. The unit tests check training-prompt uniqueness and held-out case IDs. This controlled experiment tests whether more varied conditioning improves answer generation; because all examples concern a small set of project facts, it is not evidence of broad language understanding. See [latest TinyLM workflow](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43749808243).


## Follow-up diagnosis: context-window truncation

The 48-example / 16-question run (workflow [43749808243](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43749808243)) completed successfully, but answer accuracy was 0/16 both with and without memory. The literal gold answer appeared in retrieved evidence for 13/16 questions (81.25%). The training response loss fell from 3.4140 to 0.3415, but generated answers remained incoherent.

Inspection found a concrete inference-path issue: the previous prompt put long explanatory headers and memory before the question, then truncated from the left to fit a 128-byte context. This could remove the memory evidence. The inference path now uses a compact M:/Q: prompt builder, prioritizes the question, allocates remaining bytes to retrieved evidence, and reserves up to 32 context positions for answer generation so evidence does not immediately slide out of the context. Both chat and QA evaluation use the same builder. Unit tests assert that evidence and question fit together and that generation space is reserved.

This is a prompt/context correctness fix, not evidence of improved model capability. The next completed CPU run must establish whether memory-conditioned answers improve; if not, the next bottleneck is the tiny byte-level model and its SFT data/optimization, not merely prompt truncation.

## Aligned prompt/SFT rerun (16 November 2026)

Workflow: [run 43751361173](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43751361173). The evaluator now builds both memory and no-memory prompts with the same compact context-budgeted helper used by chat. The 48 SFT examples were rewritten to the canonical `M:... / Q:... / A:` format, and the SFT encoder now preserves the memory and question while reserving room for the target answer. Unit tests verify evidence/question preservation in training and shared prompt construction in evaluation.

| Measurement | Result |
|---|---:|
| SFT response-only loss | 3.4277 to 0.2882 |
| Held-out synthetic QA without memory | 0/16 exact answer matches |
| Held-out synthetic QA with memory | 0/16 exact answer matches |
| Gold answer present in retrieved evidence | 13/16 (81.25%) |
| WikiText-2 held-out test perplexity | 11.4529 per byte |

The rerun passed end-to-end, but generated answers remained incoherent and exact-match accuracy did not improve. The SFT loss decreased, but that is not evidence of generalization. This establishes that correcting prompt alignment alone is insufficient for this byte-level 445,952-parameter model. The next useful experiment should change the modeling/data setup rather than simply add more optimization steps: a subword tokenizer and a more representative instruction dataset would address major limitations, while preserving a strict held-out set and no-memory/memory comparison. Retrieval also remains imperfect: three of sixteen questions did not retrieve evidence containing the literal expected answer. No claim of useful memory-conditioned generation is supported by this run.


## Learned tokenizer comparison (CPU experiment added)

A second tokenizer path now trains deterministic byte-level BPE merges from the WikiText-2 training split only. It starts with all 256 byte values, learns up to 256 merge operations (target vocabulary 512), and stores the merge table in the model checkpoint. No tokenizer model or vocabulary is downloaded. Validation and test text are never used to fit merges.

The BPE tokenizer is integrated through pretraining, answer-only supervised tuning, memory-prompt construction, chat inference, and held-out language-model evaluation. The paired CPU workflow keeps the Transformer size, context length, optimizer settings, training steps, instruction examples, and held-out QA cases aligned with the byte-token baseline. It reports token/byte ratios and held-out loss normalized by both token and byte counts. This is a tokenizer/context-efficiency experiment; lower token counts or training loss alone do not establish better answers.

New tests cover Unicode round-tripping, deterministic merge training, compression on repeated text, serialization, and preserving both memory evidence and the question through BPE prompt/SFT encoding.

### Completed paired result (16 November 2026)

Full CPU workflow: [run 43752163491](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43752163491). The unit-test workflow also passed: [run 43752167169](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43752167169).

| Measurement | Byte baseline | Learned BPE |
|---|---:|---:|
| Vocabulary size | 256 | 512 |
| Model parameters | 445,952 | 478,720 |
| Training tokens per input byte | 1.000 | 0.627 |
| Held-out test cross-entropy (nats/byte, normalized) | 2.4382 | 2.1735 |
| Held-out byte-normalized perplexity (exp of nats/byte) | 11.4529 | 8.791 |
| SFT response-only loss, step 1 to 2,000 | 3.4277 to 0.2882 | 4.9368 to 0.2490 |
| Held-out synthetic QA without memory | 0/16 | 0/16 |
| Held-out synthetic QA with memory | 0/16 | 0/16 |
| Gold answer present in retrieved evidence | 13/16 (81.25%) | 13/16 (81.25%) |

The BPE model uses about 37.3% fewer tokens per training byte and has lower held-out cross-entropy when normalized by original text bytes (about 10.9% lower than the byte baseline). This is a useful language-modeling/context-efficiency signal, although the BPE model has a larger embedding vocabulary and therefore 32,768 more parameters. The per-token perplexity values are not directly comparable across tokenizers; the byte-normalized loss is the relevant comparison here.

Crucially, the better language-modeling metric did **not** translate into better memory QA: both tokenizers scored 0/16 with and without retrieved memory, and retrieval evidence coverage remained 13/16. BPE generations contain more recognizable fragments than the byte model but are still not reliable answers. The experiment therefore supports keeping BPE as the more efficient tokenizer for the next iteration, not claiming that a useful memory assistant has been built. The next bottleneck is instruction-following/generalization and the tiny training setup, alongside the three retrieval misses. The model remains randomly initialized; no pretrained weights or hosted generation APIs were used.


## Corrected full byte/BPE rerun (16 November 2026)

The corrected full CPU workflow [43753114964](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43753114964) completed successfully on commit `bb7c144058cac0804765179b42674940240881a6`. It passed the full test suite, smoke training/inference, WikiText-2 preparation, random-initialized pretraining, held-out evaluation, both instruction-tuning stages, both memory QA evaluations, and artifact upload. The 30-day artifact is `tiny-memory-lm-wikitext2-pilot-43753114964` (artifact ID `14177337063`); it contains both checkpoints, training logs, held-out metrics, and per-case QA outputs.

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Vocabulary | 256 | 512 |
| Parameters | 445,952 | 478,720 |
| Pretraining steps | 1,000 | 1,000 |
| Final training loss | 2.3898 nats/token (= nats/byte) | 3.4775 nats/token |
| Held-out test cross-entropy | 2.4382 nats/byte | 2.1735 nats/byte (byte-normalized) |
| Held-out byte-normalized perplexity | 11.4529 | 8.791 (approx.) |
| SFT response-only loss, step 1 to 2,000 | 3.4277 to 0.2882 | 4.9368 to 0.2490 |
| Held-out QA without memory | 0/16 | 0/16 |
| Held-out QA with memory | 0/16 | 0/16 |
| Retrieved evidence contains literal gold answer | 13/16 (81.25%) | 13/16 (81.25%) |

BPE reduces tokens per input byte by about 37.3% and improves held-out cross-entropy normalized by original bytes by about 10.9%. Its vocabulary adds 32,768 parameters, and per-token perplexity should not be compared directly between tokenizers. The byte-normalized metric is the meaningful comparison.

The corrected evaluator separately reports the evidence-only diagnostic: retrieved text contains the literal expected answer in 13 of 16 cases for each tokenizer. The three misses are the two SQLite-database paraphrases and one next-byte-prediction paraphrase. This is evidence exposure, not a generated answer or semantic QA score. In both models, generated answer substring accuracy remains 0/16 with memory and 0/16 without memory. Sample generations remain incoherent. Thus retrieval can surface evidence for most of this tiny test set, but neither model can reliably turn it into an answer; adding memory does not currently improve generation. The earlier 10.9% language-modeling gain from BPE is real within this setup, but it is not evidence of a useful memory assistant.

The model weights were randomly initialized, training and evaluation ran on CPU, and no hosted generation API or paid compute was used. The QA set is synthetic and small, and exact substring scoring can reject semantically correct paraphrases; however, the inspected outputs are plainly not coherent answers, so the zero score reflects a genuine generation failure here. Keep retrieval and generation metrics separate in subsequent experiments.


## Latest retrieval-paraphrase rerun (16 November 2026)

Full CPU workflow: [run 43754510867](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43754510867), commit `60ac4b715d7823ddee878ccb5d168ab0877b81ac`. The workflow passed all tests, byte and BPE pretraining, held-out WikiText-2 evaluation, both instruction-tuning stages, both memory-QA comparisons, and artifact upload. Artifact: `tiny-memory-lm-wikitext2-pilot-43754510867` (ID `14177962889`; available until 15 December 2026).

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Vocabulary | 256 | 512 |
| Held-out cross-entropy (nats/byte) | 2.4382 | 2.1735 |
| Held-out byte-normalized perplexity | 11.4529 | 8.791 (approx.) |
| Generated QA without memory | 0/16 | 0/16 |
| Generated QA with memory | 0/16 | 0/16 |
| Retrieved evidence contains literal gold answer | 16/16 (100%) | 16/16 (100%) |

The three lexical retrieval misses in the previous run were addressed by conservative morphology and transparent lexical aliases. In this test set, the expected answer text was present in retrieved evidence for all 16 questions for both tokenizers, up from 13/16. This is a retrieval evidence-exposure result, not a semantic retrieval benchmark or a generated answer score.

Despite complete evidence coverage, neither instruction-tuned model produced a correct exact-substring answer in any of the 16 cases, with or without memory. The generated outputs remain incoherent. Therefore the run demonstrates a retrieval improvement, but **no measurable improvement in generated QA from adding memory**. The BPE held-out byte-normalized cross-entropy remains about 10.9% lower than the byte baseline; its larger vocabulary and different tokenization mean per-token perplexities are not directly comparable.

The full workflow ran on CPU from random initialization, without pretrained weights, hosted generation APIs, API keys, or paid compute. The 16-question synthetic evaluation is small and exact-substring scoring has limitations; the inspected generations nevertheless confirm that answer generation is currently the main bottleneck. The next research priority is to improve model capability and instruction data while keeping retrieval evidence coverage and generated-answer quality as separate metrics.

## Teacher-forced memory-use diagnostic (16 November 2026)

The memory QA evaluator now reports a second model-side measure in addition to free-generation exact-match accuracy: mean negative log-likelihood (NLL) of the expected answer when that answer is supplied token by token after the prompt. It scores the same held-out answer under both the no-memory and retrieved-memory prompts.

- A lower NLL in the memory condition means the retrieved context makes the expected answer more probable under the model.
- A negative `memory_minus_no_memory_mean_nll` is therefore a positive signal for conditional memory use.
- This remains a teacher-forced likelihood diagnostic, not generated-answer accuracy, proof of reasoning, or a replacement for held-out QA.
- The evaluator reports both measurements separately so a model cannot appear successful merely because retrieval exposed the gold answer.

The metric is implemented in `scripts/evaluate_tiny_lm_memory_qa.py` and covered by a unit test. It was validated end to end in corrected CPU workflow [43755777246](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43755777246), commit `696315007d8b7af16875a7d34ee06cafde142b36`.


### Corrected teacher-forced evaluation results

The corrected run completed successfully, including both tokenizers and artifact upload (artifact ID `14178039836`, available until 15 December 2026).

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| No-memory mean gold-answer NLL | 3.8713 | 3.1531 |
| Memory mean gold-answer NLL | 3.2912 | 2.6752 |
| Memory minus no-memory NLL | -0.5801 | -0.4778 |
| Gold-answer tokens scored | 330 | 212 |
| Retrieved evidence contains gold answer | 16/16 | 16/16 |
| Generated exact-substring QA, no memory | 0/16 | 0/16 |
| Generated exact-substring QA, memory | 0/16 | 0/16 |

For both tokenizers, the memory prompt lowered mean teacher-forced gold-answer NLL across the 16-case synthetic evaluation. This is consistent with retrieved context making the expected answer more probable to the model. It does **not** show that the model can produce the answer: free generation remained 0/16 in both conditions. Nor does the small synthetic evaluation establish statistical significance or generalization. Treat this as a preliminary signal of conditional context sensitivity, not successful long-term-memory QA.

The run's unit tests, training, held-out evaluation, instruction tuning, memory comparisons, and artifact upload all completed successfully on CPU. The weights remained randomly initialized before WikiText-2 training; no pretrained checkpoint, hosted generation API, API key, or paid compute was used.


## Irrelevant-memory control experiment (17 November 2026)

The evaluator now compares three matched prompt conditions: no memory, retrieved relevant memory, and a cross-case irrelevant-memory distractor. The distractor is the first memory from another evaluation case whose memory list differs from the current case. This is a simple control, not a guarantee of semantic unrelatedness: distractors can share topics or wording with the target question.

The full CPU workflow [43756805969](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43756805969) completed successfully on commit `8ed97bf314286943fa04675c9d15865071a7e5fd`. Both the full model workflow and unit-test workflow passed. The uploaded artifact is [14178473372](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43756805969/artifacts/14178473372), retained for 30 days.

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| No-memory mean gold-answer NLL | 3.8713 | 3.1531 |
| Relevant-memory mean gold-answer NLL | 3.2912 | 2.6752 |
| Irrelevant-memory mean gold-answer NLL | 3.9015 | 3.7115 |
| Relevant minus no-memory NLL | -0.5801 | -0.4778 |
| Irrelevant minus no-memory NLL | +0.0302 | +0.5584 |
| Generated QA, no memory | 0/16 | 0/16 |
| Generated QA, relevant memory | 0/16 | 0/16 |
| Generated QA, irrelevant memory | 0/16 | 0/16 |
| Gold answer present in retrieved evidence | 16/16 | 16/16 |

The relevant-memory prompt lowers teacher-forced gold-answer NLL for both tokenizers, while the irrelevant-memory prompt slightly worsens it for the byte model and substantially worsens it for BPE. This pattern is consistent with the model responding to the *content relevance* of retrieved context rather than benefiting from arbitrary extra memory text. It is stronger evidence than the previous two-condition comparison, but remains preliminary: only 16 synthetic cases were used, the distractors are selected across cases rather than curated and semantically audited, and aggregate NLL alone does not establish robust memory reasoning.

Free generation still scores 0/16 in every condition. Thus this experiment does **not** show a usable QA model or successful long-term-memory assistant. Retrieval exposes the gold answer in all 16 cases, but the model cannot generate it reliably. Keep this distinction explicit: retrieval supplies external context; it does not update model weights or prove that the model has learned persistent memories.

The experiment used randomly initialized weights, CPU-only GitHub Actions, no hosted generation API, no API key, and no paid compute. The next experiment should improve the control design before making stronger claims: create manually checked, topic-matched but answer-irrelevant distractors, report per-question paired NLL deltas and uncertainty, and expand the held-out set. Do not increase model size or compute until the evaluation is more diagnostic.


## Curated irrelevant-memory control (16 November 2026)

Full CPU workflow: [run 43940680500](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43940680500), on commit `ffb6b81`. The 16 held-out QA cases now include manually selected, topic-matched facts that are intended to be irrelevant to each question's expected answer. This replaces the weaker cross-case distractor fallback for these cases. The complete workflow passed tests, random-initialized byte and BPE training, held-out WikiText-2 evaluation, instruction tuning, memory QA comparisons, and artifact upload. Artifact [14261809338](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43940680500/artifacts/14261809338) contains the checkpoints, logs, metrics, and per-case outputs.

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| No-memory mean gold-answer NLL | 3.8713 | 3.1531 |
| Relevant-memory mean gold-answer NLL | 3.2912 | 2.6752 |
| Curated irrelevant-memory mean gold-answer NLL | 3.8674 | 3.7695 |
| Relevant minus no-memory NLL | -0.5801 | -0.4778 |
| Curated irrelevant minus no-memory NLL | -0.0038 | +0.6165 |
| Generated QA, no memory | 0/16 | 0/16 |
| Generated QA, relevant memory | 0/16 | 0/16 |
| Generated QA, curated irrelevant memory | 0/16 | 0/16 |
| Gold answer present in retrieved evidence | 16/16 | 16/16 |

The relevant-memory prompt lowers teacher-forced gold-answer NLL for both tokenizers. The curated distractor has almost no aggregate effect for the byte model and substantially worsens NLL for BPE. To inspect the paired case-level effects, I computed each condition's NLL minus the same question's no-memory NLL, then used 20,000 seeded percentile-bootstrap resamples of the 16 paired deltas:

| Paired NLL delta vs no memory | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Relevant memory: mean delta | -0.5801 | -0.4778 |
| Relevant memory: cases improved / worsened | 14 / 2 | 15 / 1 |
| Relevant memory: exploratory 95% bootstrap interval | [-0.9419, -0.2486] | [-0.6960, -0.2663] |
| Curated irrelevant memory: mean delta | -0.0038 | +0.6165 |
| Curated irrelevant memory: cases improved / worsened | 8 / 8 | 2 / 14 |
| Curated irrelevant memory: exploratory 95% bootstrap interval | [-0.3452, +0.3447] | [+0.3912, +0.8483] |

These paired results strengthen the observation that relevant memory makes the gold answer more probable in this small test set: 14/16 byte cases and 15/16 BPE cases improved, and both exploratory bootstrap intervals are below zero. The curated irrelevant control is less consistent across tokenizers: it has no clear aggregate effect for byte (interval spans zero), but worsens gold-answer likelihood for BPE (14/16 cases, interval above zero). Bootstrap intervals over only 16 synthetic examples are descriptive and should not be treated as broad-population statistical proof. This is a more controlled and topic-matched comparison than the earlier cross-case distractor test, but it remains a small diagnostic.

Generated-answer accuracy remains 0/16 in all conditions, despite 16/16 retrieval evidence coverage. Teacher-forced NLL measures how probable the gold answer is when supplied as the target; it is not free-generation accuracy, reasoning ability, or evidence that memories have been written into model weights. Persistent memory remains external SQLite retrieval context. The model is still not a useful QA assistant.

The run used random initialization, CPU-only GitHub Actions, no pretrained weights, no hosted generation API, no API keys, and no paid compute. A follow-up full workflow on the fixture-corrected branch head is running separately; the source/test correction does not change these evaluator or dataset results.


## Unseen-fact memory generalization evaluation (17 November 2026)

The earlier 16-question held-out set paraphrases facts that also occur in the SFT demonstrations. It is useful for measuring paraphrase transfer, but it does not test whether the model can use memory to answer about facts absent from instruction tuning. A second diagnostic set, `examples/tiny_lm_memory_qa_unseen_facts.jsonl`, now addresses that limitation.

The new set contains 16 questions over eight separate fact values (archive export format, sync interval, audit retention, backup region, embedding-cache dimensionality, conflict policy, memory identifier format, and retrieval similarity threshold). Each fact has two question phrasings and one manually selected distractor. The expected values and complete memory facts are checked to be absent from the SFT demonstration text. The workflow evaluates the same byte and BPE instruction-tuned checkpoints on both the original paraphrase set and this unseen-fact set, using no-memory, relevant-memory, and curated irrelevant-memory conditions.

This is a diagnostic of memory-conditioned fact use on unseen values, not a broad benchmark. It is still synthetic, small, and created by the project author; exact substring scoring and lexical retrieval have known limitations. ### Completed unseen-fact result (17 November 2026)

The full CPU workflow [43943240352](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43943240352) completed successfully on branch `research/long-term-memory-starter`, commit `a79a8285d84bd1be955efe80e36f4142a704db8c`. All tests, byte and BPE training, held-out language-model evaluation, instruction tuning, both original and unseen-fact QA evaluations, and artifact upload passed. The 30-day artifact is [14264341714](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43943240352/artifacts/14264341714); it includes checkpoints, logs, metrics, and per-case outputs.

| Unseen-fact measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Held-out test cross-entropy (nats/byte) | 2.4382 | 2.1735 |
| No-memory mean gold-answer NLL | 5.9945 | 7.4471 |
| Relevant-memory mean gold-answer NLL | 6.1474 | 7.6646 |
| Irrelevant-memory mean gold-answer NLL | 6.1376 | 7.7776 |
| Relevant minus no-memory NLL | +0.1529 | +0.2174 |
| Irrelevant minus no-memory NLL | +0.1431 | +0.3304 |
| Generated QA, no memory | 0/16 | 0/16 |
| Generated QA, relevant memory | 0/16 | 0/16 |
| Generated QA, curated irrelevant memory | 0/16 | 0/16 |
| Retrieved evidence contains gold answer | 16/16 | 16/16 |
| Retrieval-only evidence contains gold answer | 16/16 | 16/16 |

**The unseen-fact result does not show a memory benefit.** In contrast with the earlier paraphrase set (where relevant memory lowered teacher-forced NLL), adding the relevant memory slightly increased mean gold-answer NLL for both tokenizers. The irrelevant-memory control also increased NLL. Thus the model is not demonstrating that it can use retrieved context to make novel fact values more probable, much less generate them. All free-generation outputs failed exact normalized answer matching; inspection shows incoherent or unrelated continuations.

Because the 16 questions are two paraphrases of each of only eight fact values, I also averaged the paired relevant-minus-no-memory NLL deltas within each fact, then bootstrapped the eight fact clusters (20,000 seeded percentile resamples). The mean cluster delta was +0.1529 nats for byte (3/8 facts improved, 5/8 worsened; exploratory 95% interval [-0.0108, +0.3003]) and +0.2174 for BPE (1/8 improved, 7/8 worsened; interval [+0.0156, +0.4294]). These intervals are descriptive only: eight synthetic fact clusters are far too few to support broad statistical claims. Resampling the 16 questions as if independent would overstate the effective sample size.

The 16/16 retrieval score only says that lexical search exposed text containing the answer. The retrieval-only evidence baseline is not a generated answer and is not evidence of model reasoning. The earlier 16-case paraphrase benchmark and this unseen-fact benchmark answer different questions: the former showed a positive teacher-forced context-sensitivity signal; this latter test fails to reproduce it on facts excluded from SFT. This contrast is the main result and should not be averaged away.

No training examples or model weights were changed to run this evaluation. The models were randomly initialized, trained on WikiText-2 using CPU GitHub Actions, and instruction-tuned on the same small synthetic SFT set; no pretrained weights, hosted generation API, API key, or paid compute were used. The next research step should improve the model's ability to condition on retrieved facts and expand the fact-clustered held-out evaluation before scaling model size or compute. Persistent memory remains external SQLite retrieval context, not weight-based continual learning.


## Memory-grounding SFT data and generated-answer overlap metric (17 November 2026)

A follow-up experiment adds `scripts/generate_memory_sft.py`, which deterministically creates 64 additional prompt/response examples from 16 synthetic training facts, each paired with four question phrasings. The records use the canonical `M:...\nQ:...\nA:` prompt format and train short answers copied from the supplied memory. The workflow builds this corpus at run time, combines it with the existing 48 demonstrations, and uses the resulting 112 examples for both byte and BPE instruction-tuning runs. The generator's values are checked against the eight unseen-fact clusters to prevent exact answer or full-memory overlap.

The QA evaluator now reports mean token-set F1 alongside exact normalized answer-substring accuracy. This makes partial lexical overlap visible when a generated response is not an exact match; it is still not semantic judging and can reward irrelevant token overlap. Teacher-forced NLL remains a separate metric. A test covers exact, partial, unrelated, and empty outputs.

The first generator test run exposed an assertion that mistakenly expected the two literal characters `\\n`; it was corrected to assert an actual newline. The corrected full tiny-LM workflow is still running at the time of this update. No result is yet available for whether the expanded SFT data improves unseen-fact generation. The current evaluation continues to use the original eight held-out fact clusters, so this is an exploratory change rather than a new independent benchmark.



## Completed memory-grounding SFT follow-up (17 November 2026)

The full CPU workflow [43945536520](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43945536520) completed successfully on source commit `d69dc399c2f3ddbf3d20b03ba33ef859b5b21501`. The full test suite, smoke test, byte and BPE pretraining, held-out corpus evaluation, 112-example memory-grounding SFT, both familiar-paraphrase QA evaluations, both unseen-fact evaluations, and artifact upload all passed. The [30-day workflow artifact](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/43945536520/artifacts/14265825111) contains checkpoints, logs, and per-case metrics.

| Measurement | Byte tokenizer | Learned BPE |
|---|---:|---:|
| Held-out WikiText-2 cross-entropy (nats/byte) | 2.4382 | 2.1735 |
| Familiar paraphrases: generated exact answers with relevant memory | 0/16 | 2/16 |
| Familiar paraphrases: mean token-set F1 with relevant memory | 0.0469 | 0.2703 |
| Familiar paraphrases: relevant-memory minus no-memory gold-answer NLL | -0.5519 | -0.8587 |
| Familiar paraphrases: irrelevant-memory minus no-memory gold-answer NLL | +0.0540 | +0.5898 |
| Unseen facts: generated exact answers with relevant memory | 0/16 | 0/16 |
| Unseen facts: mean token-set F1 with relevant memory | 0.0313 | 0.0295 |
| Unseen facts: relevant-memory minus no-memory gold-answer NLL | -0.1164 | +0.0136 |
| Unseen facts: irrelevant-memory minus no-memory gold-answer NLL | +0.0484 | -0.1007 |
| Retrieved evidence contains gold answer (familiar / unseen) | 16/16 / 16/16 | 16/16 / 16/16 |

The expanded SFT data improved some *diagnostic signals* on the familiar paraphrase set, especially for BPE: relevant context lowered teacher-forced answer NLL, and two generated answers passed exact substring matching. However, this is not a successful general memory QA result. Byte exact generation remained 0/16; BPE was 2/16, and most BPE generations were still malformed or incomplete. Token F1 is a lexical overlap diagnostic, not semantic answer quality.

Crucially, the familiar paraphrase cases are not independent of the training facts in the same way as the unseen-fact set. On the eight unseen fact clusters (16 phrasings), both tokenizers generated 0/16 exact answers with relevant memory. Byte relevant memory lowered mean gold-answer NLL by 0.1164 nats, while BPE raised it by 0.0136 nats. The paired cluster bootstrap (two phrasings averaged within each of eight fact clusters; 20,000 seeded percentile resamples) gives byte 4/8 clusters improved and exploratory 95% interval [-0.4373, +0.1366], and BPE 5/8 improved with interval [-0.3097, +0.4014]. These wide intervals are descriptive only; eight synthetic clusters are insufficient for broad inference.

Retrieval again found answer-containing evidence in all 16 familiar and all 16 unseen cases for both tokenizers. That establishes successful evidence exposure by the lexical retriever, not model comprehension. The retrieval-only baseline can expose the literal answer because the answer is in the retrieved record; it must not be conflated with generated QA. The byte-model chat smoke output was `arndsout dse desther`, illustrating that the model remains far from coherent assistant behavior.

**Conclusion:** the new SFT data increases familiar-set teacher-forced context sensitivity and gives BPE a small number of exact familiar-set hits, but it does not establish generalization to new memory values or reliable free generation. The model still does not qualify as a useful assistant, and memory is still supplied as external SQLite retrieval context rather than learned as persistent model weights. The experiment used random initialization, WikiText-2, CPU GitHub Actions, no pretrained weights, no hosted generation API, no API keys, and no paid compute. The next improvement should target the learning objective and generation protocol (including explicit answer-boundary/stop behavior and evidence-grounded answer formatting), then test on a larger fact-clustered holdout; scaling model size or compute is not justified by these results alone.


## Explicit answer-boundary stopping (17 November 2026)

The SFT pipeline appends a newline after each target response, but the inference paths previously generated a fixed number of tokens without stopping at that learned boundary. This could let the model continue with unrelated text after a short answer and lower exact-match scores. `TinyLanguageModel.generate` now accepts an optional token-sequence stop marker. The chat CLI passes the active tokenizer's encoding of newline, and the QA evaluator stops when decoded generated text reaches newline. Using a token sequence rather than assuming one token keeps the interface compatible with byte and BPE tokenizers.

A unit test forces a two-token stop marker and asserts that generation ends immediately after the marker. This is an inference/protocol correction only: it does not change model weights or establish improved answer quality. The next CPU run must compare outputs and metrics against the completed 112-example SFT baseline.


## Expanded independent unseen-fact benchmark (17 November 2026)

The unseen-fact fixture now contains **64 questions over 32 independent synthetic fact clusters**, with two paraphrases per fact. Each cluster has one answer-bearing memory and a manually authored answer-irrelevant distractor. The 32 values and memory statements are checked for exact textual overlap with both the 48-example demonstration file and the 64 generated memory-grounding SFT examples. This is a larger and more defensible diagnostic than the earlier eight-cluster holdout, although the facts remain synthetic and manually authored.

The QA evaluator now reports paired memory-minus-no-memory deltas after averaging paraphrases within each fact cluster. It uses a seeded percentile bootstrap that resamples clusters, not individual paraphrase rows, and reports the 95% interval, cluster count, and positive/negative/zero delta counts for:
- exact generated-answer match rate;
- token-set F1 (lexical overlap only);
- teacher-forced gold-answer mean NLL.

For NLL, a negative delta favors relevant memory; for exact-match accuracy and token F1, a positive delta favors relevant memory. These are separate diagnostics and should not be collapsed into one score. The intervals are exploratory and only generalize to this constructed set, not to real user memories.

The new dataset and cluster-aware evaluator have unit tests for 32 clusters, two paraphrases per cluster, deterministic bootstrap behavior, and equal weighting of clusters rather than paraphrase counts. The expensive training workflow is now opt-in for pull-request activity and can be started manually; ordinary unit tests continue to run on branch pushes and pull requests. A full CPU training rerun is still required before reporting metrics for this expanded fixture.

