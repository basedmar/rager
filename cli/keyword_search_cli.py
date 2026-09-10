import argparse
from helper import *
from inverted_index import *
import sys

def exit():
    sys.exit()

def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    subparsers.add_parser("build", help="build the inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    

    stop_words = get_stop_words("/home/kkmarmar/rag-search-engine/data/stopwords.txt")

    args = parser.parse_args()
    movies = load_mov("/home/kkmarmar/rag-search-engine/data/movies.json")
    count = 0
    inv_index = InvertedIndex()

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