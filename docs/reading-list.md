# Reading list: long-term memory for LLM agents

1. **MemGPT: Towards LLMs as Operating Systems** — hierarchical context management and paging between working and external memory. https://sky.cs.berkeley.edu/project/memgpt/
2. **LongMemEval** — benchmark for long-term conversational memory, including extraction, multi-session reasoning, temporal reasoning, and abstention. https://redis.github.io/redis-ai-research-public/longmemeval-agent-memory/
3. **LongMemEval-V2** — long web-agent trajectories, state tracking, and workflow knowledge. https://huggingface.co/papers/2605.12493
4. **Letta research** — memory models, continual learning, and evaluation of agents that learn from experience. https://www.letta.com/research/

## Questions while reading
- What is stored: raw turns, chunks, atomic facts, summaries, entities, relations, or trajectories?
- Who decides what to write, update, consolidate, and forget?
- How are contradictions and time-dependent facts represented?
- Does retrieval return evidence with provenance?
- Are memory quality and answer-model quality evaluated separately?
- What are latency, token, and storage costs?

## API references
- Gemini free-tier and pricing: https://ai.google.dev/gemini-api/docs/pricing
- Gemini generateContent: https://ai.google.dev/api/generate-content
- Gemini embeddings: https://ai.google.dev/gemini-api/docs/embeddings
