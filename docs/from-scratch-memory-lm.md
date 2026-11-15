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

The base model is only a next-byte predictor. The optional second stage trains on JSONL prompt/response examples and masks the prompt bytes from the loss, so the model is optimized to produce responses rather than copy prompts. The synthetic demonstration set is for pipeline development, not a substitute for a licensed, diverse conversational corpus. The expanded set now contains 48 prompt/response examples across eight project facts and six prompt styles per fact; it remains deliberately small.

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

The very low instruction-tuning loss alongside zero QA accuracy is a warning that the earlier eight-example demonstration set was being memorized or otherwise overfit, not that the model had learned general instruction following. The next controlled experiment expands training to 48 prompt/response examples over eight facts and evaluates on 16 separately phrased questions. This is still synthetic and small; the result must be judged by held-out answers, not training loss. The evidence-retrieval score is reported separately from generation quality.

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
