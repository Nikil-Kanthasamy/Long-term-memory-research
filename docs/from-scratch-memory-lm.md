# From-scratch language model with long-term memory

## Goal

Build a small, inspectable decoder-only Transformer whose weights are initialized randomly and trained locally, then connect it to the existing SQLite memory store. This is a research prototype, not a claim that a small CPU-trained model can match a frontier assistant.

## What is genuinely from scratch

- No pretrained model weights are loaded.
- No hosted generation API is called by the tiny-model training or inference scripts.
- The tokenizer is a fixed UTF-8 byte vocabulary (256 IDs), so there is no downloaded pretrained tokenizer.
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
