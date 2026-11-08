"""Command-line interface."""
from .config import db_path,load_env_file,top_k
from .llm import generate
from .memory import answer,remember,retrieve
from .store import MemoryStore
import argparse

def main():
    load_env_file()
    parser=argparse.ArgumentParser(description="Long-term memory research baseline")
    sub=parser.add_subparsers(dest="command",required=True)
    add=sub.add_parser("add"); add.add_argument("--user",required=True); add.add_argument("--text",required=True)
    search=sub.add_parser("search"); search.add_argument("--user",required=True); search.add_argument("--query",required=True)
    ask=sub.add_parser("ask"); ask.add_argument("--user",required=True); ask.add_argument("--question",required=True)
    gen=sub.add_parser("generate"); gen.add_argument("--user",required=True); gen.add_argument("--prompt",required=True)
    forget=sub.add_parser("forget"); forget.add_argument("--user",required=True); forget.add_argument("--memory-id",required=True,type=int)
    args=parser.parse_args(); store=MemoryStore(db_path())
    if args.command=="add":
        turn,count=remember(store,args.user,args.text); print(f"Saved turn {turn}; extracted {count} memory item(s).")
    elif args.command=="search":
        for item in retrieve(store,args.user,args.query,top_k()):
            print(f"[{item['id']}] ({item['category']}, score={item['score']:.3f}) {item['content']}")
    elif args.command=="ask":
        print(answer(store,args.user,args.question,top_k()))
    elif args.command=="generate":
        memories=retrieve(store,args.user,args.prompt,top_k())
        context="\n".join("- "+m["content"] for m in memories)
        print(generate(f"Relevant long-term memories:\n{context or '(none)'}\n\nTask:\n{args.prompt}"))
    elif args.command=="forget":
        print("Deleted." if store.delete_memory(args.user,args.memory_id) else "Memory not found.")

if __name__=="__main__":
    main()
