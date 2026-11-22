# Joint-span memory-key representation correction

## Finding

The joint-span selector previously built its memory keys with:

```python
model.token_embedding(input_ids[:, 2:memory_end])
```

The Tiny LM has a separate `position_embedding` and a causal Transformer. Raw token embeddings therefore discard both absolute position and contextual information before the span head sees each memory byte. Since this is a byte tokenizer, repeated occurrences of the same byte receive exactly the same raw key vector. This is a particularly poor representation for locating short extractive answers among repeated characters or substrings.

## Change

The joint-span path now uses the contextualized, position-aware hidden states already computed by the Transformer:

```python
hidden[:, 2:memory_end, :]
```

The query state and memory keys now come from the same normalized Transformer representation space. A small unit test covers the memory-key slicing helper.

## Experimental implications

- This is a representation correction, not a decoding tweak.
- Existing joint-span heads were trained against raw token embeddings and are not valid checkpoints for this changed representation. Retrain each seed from the same copy-enabled base checkpoint before comparing.
- Do not compare a retrained result against the old checkpoint as if only the evaluation protocol changed.
- Keep the held-out development fixture for model/decoding selection and preserve the 64-question unseen-facts fixture for final evaluation only.
- The previous span-width sweep found no benefit from positive width penalties; that result applies to the old representation and should not be treated as a result for this corrected model.

## Limitations

The change is a well-motivated hypothesis based on the architecture, not evidence of improved accuracy. The unit test verifies the key representation helper but does not establish end-to-end quality. A CPU multi-seed retraining run and the held-out development evaluation are still required. GitHub Actions status must be checked independently; no CI success is claimed here.
