import argparse
from lib.semantic_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("verify")
    verify_embed = subparsers.add_parser("verify_embeddings")
    user_embed = subparsers.add_parser("embed_query")
    user_embed.add_argument("query", type=str)
    embed = subparsers.add_parser("embed_text")
    embed.add_argument("text", type=str)
    search = subparsers.add_parser("search")
    search.add_argument("query", type=str)
    search.add_argument("--limit", type=int, nargs="?", default=5)
    
    args = parser.parse_args()

    match args.command:
        case "search":
            res = searcher(args.query, args.limit)
            for num, doc in enumerate(res):
                print(f"({num}) {doc["title"]} ({doc["score"]})\n {doc["description"]}")
        case "embed_query":
            embed_user_query(args.query)
        case "verify_embeddings":
            verify_embedding()
        case "embed_text":
            text = args.text
            embed_text(text)
        case "verify":
            verify_model()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()