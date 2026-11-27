# Independent LoCoMo turn-retrieval validation of frozen BM25 stopword ablation

## Reproduction and data provenance

- Workflow: [45929766151](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45929766151)
- Unit tests: [45929766202](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45929766202)
- Branch: `research/long-term-memory-starter`
- Commit: `6dd41ca58b3d55dcb718c72ac141f6681576c19d`
- Artifact: `locomo-independent-retrieval-45929766151` (artifact ID 15168326020; metrics only; retained 30 days).
- Dataset: official [Snap Research LoCoMo repository](https://github.com/snap-research/locomo), pinned to source commit `cbfbc1dba6bc53d00625212a0f22d55ffee7c1fc`; data file `data/locomo10.json).
- The source README describes ten multi-session conversations and QA annotations with evidence dialog IDs. The CI workflow downloads the pinned dataset at run time; neither the dataset nor dialogue text is committed or uploaded in the artifact.
- Conversations: 10. QA items excluded as adversarial/unanswerable: 446. Answerable QA items: 1,540.
- Eligible retrieval questions: 1,531 (99.42% of answerable items). Nine answerable items were excluded because five had evidence IDs not present in the flattened dialogue turns and four lacked evidence annotations.
- Per question, the corpus contains 369–689 dialogue turns.
- Relevance: a dialogue turn is relevant exactly when its `dia_id` appears in the QA item's annotated `evidence` list.
- Scope: turn-level evidence retrieval only. No generated answers or LLM-as-judge accuracy are measured.

## Frozen comparison protocol

The stopword list was frozen from the preceding LongMemEval experiment and applied without modification. No LoCoMo-specific stopword tuning was performed.

- **Control:** BM25, Okapi `k1=1.5`, `b=0.75`, unchanged tokenizer.
- **Ablation:** same BM25 implementation and parameters, removing the fixed LongMemEval stopword list from both query and document terms.
- **Comparator:** token-set Jaccard.
- **Cutoffs:** 1, 3, 5, and 10.
- **Uncertainty:** paired bootstrap resampling of the ten conversations as clusters (20,000 resamples; seed 20261127). All questions from each sampled conversation remain together. This is preferable to treating the 1,531 questions as independent observations, but ten clusters is still a small basis for interval estimation.

## Aggregate retrieval results

| Metric | Jaccard | Default BM25 | Frozen stopword ablation | Ablation vs BM25 |
|---|---:|---:|---:|---:|
| Recall@5 | 18.58% | 41.20% | 44.79% | +3.59 pp |
| Hit@5 | 20.90% | 45.72% | 49.44% | +3.72 pp |
| MRR | 14.97% | 34.96% | 38.32% | +3.36 pp |
| Recall@10 | 24.07% | 48.49% | 52.45% | +3.97 pp |
| Hit@10 | 27.17% | 53.63% | 57.81% | +4.18 pp |

The paired conversation-cluster bootstrap 95% intervals for the ablation-minus-default change were:

| Metric | Mean change | Cluster-bootstrap 95% CI |
|---|---:|---:|
| Recall@5 | +3.59 pp | [+1.81, +5.40] pp |
| Hit@5 | +3.72 pp | [+1.72, +5.70] pp |
| MRR | +3.36 pp | [+2.32, +4.51] pp |
| Recall@10 | +3.97 pp | [+2.75, +5.13] pp |
| Hit@10 | +4.18 pp | [+2.68, +5.56] pp |

## Results by question type

| Question type | Questions | Default BM25 Recall@5 | Ablation Recall@5 | Change |
|---|---:|---:|---:|---:|
| multi-hop | 89 | 16.08% | 19.94% | +3.87 pp |
| open-domain | 841 | 50.63% | 54.50% | +3.86 pp |
| single-hop | 281 | 12.13% | 13.90% | +1.77 pp |
| temporal | 320 | 48.93% | 53.31% | +4.37 pp |

MRR change was positive for all four question types: multi-hop +0.00 pp (effectively flat), open-domain +2.83 pp, single-hop +2.33 pp, and temporal +6.59 pp. The multi-hop MRR change is effectively zero despite its modest Recall@5 increase.

## Paired question and conversation outcomes

Across 1,531 matched QA items:

| Metric | Improved questions | Regressed questions | Unchanged |
|---|---:|---:|---:|
| Recall@5 | 109 | 49 | 1,373 |
| Hit@5 | 100 | 43 | 1,388 |
| MRR | 710 | 352 | 469 |
| Recall@10 | 112 | 42 | 1,377 |
| Hit@10 | 100 | 36 | 1,395 |

At the conversation level, Recall@5 improved in 9 of 10 conversations and regressed in 1; MRR improved in all 10 conversations. Thus the gain is not confined to a single conversation, although question-level changes are still concentrated in a minority of QA items.

## Interpretation

This is an independent-dataset replication of the direction of the LongMemEval result: the same frozen stopword list improves Recall@5, Hit@5, and MRR on LoCoMo. It provides stronger evidence that removing the fixed common-word list can help this BM25 setup beyond the benchmark on which the list was selected.

It does **not** establish a universal BM25 improvement. LoCoMo contains only ten conversations, some answerable items lack usable evidence annotations, and the cluster-bootstrap intervals are based on only ten independent conversation units. The retrieval labels measure annotated evidence-ID recovery, not whether retrieved context is sufficient to answer a question. LoCoMo's category mix and corpus structure also differ from LongMemEval, so absolute scores should not be compared across the two datasets.

## Research decision and next step

Keep default BM25 as the stable control and retain the frozen stopword condition as a promising research variant. Do not silently replace the default. Next, test on a third independent conversational-memory dataset or a larger set of held-out conversations, keep the list fixed, and report cluster-aware uncertainty and subgroup regressions. End-to-end QA must be evaluated separately before making claims about answer quality.

## Citation

Maharana et al. (2024), *Evaluating Very Long-Term Conversational Memory of LLM Agents*, arXiv:2402.17753. Dataset and evidence annotations: [snap-research/locomo](https://github.com/snap-research/locomo).
