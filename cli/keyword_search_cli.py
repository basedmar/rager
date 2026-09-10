import argparse
from helper import *
from inverted_index import *
import sys

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="build the inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for {args.query}")
            results = search(args.query)
            for res, count in enumerate(results):
                print(f"{count}. {res} {count}")
        case "build":
            build()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()