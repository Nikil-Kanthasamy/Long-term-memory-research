"""Run the locally trained tiny LM with external SQLite long-term memory."""
import argparse

import torch

from src.ltm.config import db_path
from src.ltm.store import MemoryStore
from src.ltm.tiny_lm import ByteTokenizer, TinyLanguageModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/tiny-memory-lm/best.pt")
    parser.add_argument("--user", required=True, help="Memory namespace/user ID")
    parser.add_argument("--question", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--remember", action="store_true",
                        help="Explicitly store the question as a raw turn and memory")
    parser.add_argument("--database", default=db_path())
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = TinyLanguageModel.from_checkpoint(checkpoint, map_location=device)
    store = MemoryStore(args.database)
    memories = store.search(args.user, args.question, limit=args.top_k, mode="lexical")
    memory_text = "\n".join(f"- {item['content']}" for item in memories)
    prompt = (
        "Long-term memory (use only when relevant):\n"
        f"{memory_text or '(no matching memories)'}\n\n"
        f"User: {args.question}\nAssistant:"
    )
    tokenizer = ByteTokenizer()
    token_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if token_ids.size(1) > model.config.block_size:
        token_ids = token_ids[:, -model.config.block_size:]
    generated = model.generate(
        token_ids, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )
    response = tokenizer.decode(generated[0, token_ids.size(1):].tolist())
    print(response.strip())

    if args.remember:
        turn_id = store.add_turn(args.user, args.question)
        store.add_memory(args.user, args.question, category="conversation",
                         importance=0.4, confidence=1.0, source_turn_id=turn_id)
        print("\n[Saved this question to long-term memory]")


if __name__ == "__main__":
    main()
