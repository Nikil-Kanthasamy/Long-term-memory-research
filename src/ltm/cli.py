"""Command-line interface for the memory research baseline."""
import argparse
from .config import db_path, load_env_file, top_k
from .llm import generate
from .memory import answer, remember, retrieve
from .eval import evaluate_cases, load_cases
from .store import MemoryStore

def main():
    load_env_file()
    parser = argparse.ArgumentParser(description="Long-term memory research baseline")
    sub = parser.add_subparsers(dest="command", required=True)
    add = sub.add_parser("add"); add.add_argument("--user", required=True); add.add_argument("--text", required=True)
    search = sub.add_parser("search"); search.add_argument("--user", required=True); search.add_argument("--query", required=True)
    ask = sub.add_parser("ask"); ask.add_argument("--user", required=True); ask.add_argument("--question", required=True)
    gen = sub.add_parser("generate"); gen.add_argument("--user", required=True); gen.add_argument("--prompt", required=True)
    forget = sub.add_parser("forget"); forget.add_argument("--user", required=True); forget.add_argument("--memory-id", required=True, type=int)
    history = sub.add_parser("history"); history.add_argument("--user", required=True); history.add_argument("--query", required=True)
    timeline = sub.add_parser("timeline"); timeline.add_argument("--user", required=True); timeline.add_argument("--memory-id", required=True, type=int)
    evaluate = sub.add_parser("evaluate"); evaluate.add_argument("--user", required=True); evaluate.add_argument("--dataset", required=True); evaluate.add_argument("--mode", choices=["lexical", "semantic", "hybrid"], default="hybrid"); evaluate.add_argument("--k", type=int, default=5); evaluate.add_argument("--output")
    lme = sub.add_parser("longmemeval"); lme.add_argument("--dataset", required=True); lme.add_argument("--k", type=int, default=5); lme.add_argument("--limit", type=int); lme.add_argument("--output", default="results/longmemeval_retrieval.json"); lme.add_argument("--generate-answers", action="store_true"); lme.add_argument("--predictions", default="results/longmemeval_hypotheses.jsonl")
    args = parser.parse_args()
    store = None if args.command == "longmemeval" else MemoryStore(db_path())
    if args.command == "add":
        turn, count = remember(store, args.user, args.text)
        print(f"Saved turn {turn}; extracted {count} memory item(s).")
    elif args.command == "search":
        for item in retrieve(store, args.user, args.query, top_k()):
            print(f"[{item['id']}] ({item['category']}, score={item['score']:.3f}) {item['content']}")
    elif args.command == "ask":
        print(answer(store, args.user, args.question, top_k()))
    elif args.command == "generate":
        memories = retrieve(store, args.user, args.prompt, top_k())
        context = "\n".join("- " + m["content"] for m in memories)
        print(generate(f"Relevant long-term memories:\n{context or '(none)'}\n\nTask:\n{args.prompt}"))
    elif args.command == "forget":
        print("Deleted." if store.delete_memory(args.user, args.memory_id) else "Memory not found.")
    elif args.command == "history":
        for turn in store.search_turns(args.user, args.query, limit=top_k()):
            print(f"[turn {turn['id']}] {turn['created_at']} (score={turn['score']:.3f}) {turn['text']}")
    elif args.command == "timeline":
        for item in store.memory_history(args.user, args.memory_id):
            print(f"[{item['id']}] {item['status']} | {item['created_at']} | {item['content']}")
    elif args.command == "evaluate":
        from .llm import embed
        report = evaluate_cases(store, load_cases(args.dataset), args.user, args.mode, args.k, embed)
        rendered = __import__("json").dumps(report, indent=2)
        if args.output:
            from pathlib import Path
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(rendered)
    elif args.command == "longmemeval":
        import json
        from pathlib import Path
        from .longmemeval import run_longmemeval, write_hypotheses_jsonl
        report = run_longmemeval(args.dataset, k=args.k, limit=args.limit,
                                 generate_answers=args.generate_answers)
        hypotheses = report.pop("hypotheses")
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                          encoding="utf-8")
        if args.generate_answers:
            write_hypotheses_jsonl(hypotheses, args.predictions)
            print(f"Saved {len(hypotheses)} hypotheses to {args.predictions}")
        print(json.dumps({"metrics": report["metrics"],
                          "questions_scored": report["questions_scored"],
                          "no_gold_evidence_questions": report["no_gold_evidence_questions"],
                          "report": str(output)}, indent=2))

if __name__ == "__main__":
    main()
