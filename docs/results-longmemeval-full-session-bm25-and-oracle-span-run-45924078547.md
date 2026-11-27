# Full-session LongMemEval-S retrieval and oracle-span experiment

- Workflow: [45924078547](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45924078547)
- Workflow conclusion: success (39/39 steps; no failed steps)
- Evaluated commit: `3d2d93e3800389085e61c60f1afedfcec8aed571`
- Artifact: [tiny-memory-lm-wikitext2-pilot-45924078547](https://github.com/Nikil-Kanthasamy/Long-term-memory-research/actions/runs/45924078547/artifacts/15166610440)
- Dataset: official LongMemEval-S, 500 questions

This workflow contains two distinct evaluations: full-session retrieval over all 500 questions, and extractive span localization on a small oracle-evidence subset. They must not be conflated.

## 1. Full-session retrieval

The retrieval builder included all 500 questions and did not skip any. It ranks full conversation sessions; a session is relevant when its ID appears in that question's benchmark evidence list. Both methods had relevant evidence available in the haystack for all 500 questions.

| Metric | Lexical Jaccard | BM25 |
|---|---:|---:|
| Questions | 500 | 500 |
| Recall@1 | 9.0% | 86.2% |
| Recall@3 | 23.2% | 94.4% |
| Recall@5 | 34.6% | 96.8% |
| Recall@10 | 60.8% | 98.2% |
| MRR | 0.2299 | 0.9064 |
| Median first relevant rank | 8 | 1 |

Per-question-type Recall@5 and MRR:

| Question type | N | Jaccard R@5 | BM25 R@5 | Jaccard MRR | BM25 MRR |
|---|---:|---:|---:|---:|---:|
| knowledge-update | 78 | 38.46% | 100.00% | 0.2201 | 0.9712 |
| multi-session | 133 | 27.07% | 96.24% | 0.1790 | 0.9123 |
| single-session-assistant | 56 | 91.07% | 100.00% | 0.6570 | 1.0000 |
| single-session-preference | 30 | 6.67% | 86.67% | 0.0584 | 0.5588 |
| single-session-user | 70 | 27.14% | 98.57% | 0.2004 | 0.9370 |
| temporal-reasoning | 133 | 26.32% | 95.49% | 0.1610 | 0.8853 |

The Jaccard and BM25 implementations are lexical baselines, not semantic retrievers. BM25 IDF is computed per question's own haystack; Jaccard ties are deterministically ordered by session ID. This experiment's denominator is all 500 benchmark questions. It is a separate full-session protocol from the earlier answerable-only retrieval report, which used 470 questions; the metrics should not be compared as if they shared a denominator and evaluation setup.

## 2. Oracle-evidence extractive span localization

The span builder selected only questions for which a `has_answer=true` evidence turn contained the exact reference answer string after deterministic truncation to a 128-byte context block. It retained 20 of 500 questions (4.0%) and skipped 480.

| Eligibility outcome | Questions |
|---|---:|
| Eligible | 20 |
| Abstention | 30 |
| Answer not literal in retained evidence | 342 |
| Invalid question or answer | 32 |
| Question exceeds context | 76 |
| Total skipped | 480 |

Eligible question types: knowledge-update 4, single-session-assistant 1, single-session-user 12, temporal-reasoning 3.

Three runs used seeds 13, 29, and 47. Each used the same copy-enabled base checkpoint initialization for its seed, then fine-tuned the language model and contextual joint-span head for 600 steps. The model received a gold evidence turn, so retrieval was bypassed. Candidate ranking considered all ordered spans with start <= end over retained memory bytes; no gold answer length or length-conditioned candidate window was supplied.

| Seed | Examples | Exact joint-span accuracy | Start top-1 | End top-1 | Mean gold-span rank |
|---:|---:|---:|---:|---:|---:|
| 13 | 20 | 0.0% (0/20) | 15.0% | 25.0% | 564.00 |
| 29 | 20 | 0.0% (0/20) | 15.0% | 20.0% | 493.95 |
| 47 | 20 | 10.0% (2/20) | 5.0% | 15.0% | 510.65 |
| **Mean** | **20 per seed** | **3.33%** | **11.67%** | **20.00%** | **522.87** |

The three seeds evaluate the same 20 questions. The 60 seed-question measurements are repeated runs, not 60 independent examples. Only seed 47 localized exact spans, on two questions.

## Interpretation and limitations

1. **Session retrieval is strong for this lexical benchmark.** BM25 substantially outperformed token-set Jaccard across all reported ranks and question types. This does not establish semantic retrieval quality or end-to-end answer correctness.
2. **Exact span localization is weak.** Mean exact accuracy was 3.33% even though the model was given gold evidence, and no answer-length hint was used.
3. **The span subset is highly selected.** Only 20 questions were eligible. The exact-string and 128-byte truncation rules exclude semantic answers, multi-turn evidence, and many otherwise valid questions. The 3.33% result must not be generalized to all 500 questions.
4. **These are separate stages.** The retrieval metrics evaluate session ranking; the span metrics bypass retrieval and evaluate answer-string localization inside supplied evidence. Neither is a generated-answer or end-to-end LongMemEval score.
5. **No claim of statistical significance is made.** The small oracle subset and repeated seeds do not support a broad generalization claim.

The artifact includes `longmemeval-retrieval-summary.json`, `longmemeval-span-eval-build-summary.json`, per-seed span JSON/logs, and the constructed evaluation JSONL files. The full workflow passed and uploaded these outputs.

## Next research priorities

- Evaluate semantic answer matching that does not require the exact answer string to occur in one truncated turn.
- Preserve multi-turn evidence and measure retrieval plus answer localization jointly.
- Include abstention, temporal updates, and multi-session reasoning in the answer evaluation rather than silently filtering them away.
- Compare BM25 with a semantic or hybrid retriever under the same 500-question denominator.
- Keep per-question predictions and confidence intervals, and report generated-answer quality separately from retrieval and span localization.

## Overall conclusion

This run establishes a strong BM25 session-ranking baseline on the 500-question LongMemEval-S retrieval protocol, but it does not establish successful memory question answering. The oracle-evidence span experiment is a negative result: exact localization averaged only 3.33% on 20 highly filtered cases despite access to gold evidence. The next milestone should be a more representative end-to-end evaluation, not a stronger performance claim.
