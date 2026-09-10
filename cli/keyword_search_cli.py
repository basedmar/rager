import argparse
from helper import *
from nltk.stem import PorterStemmer
def main() -> None:
    stemmer = PorterStemmer()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")

    stop_words = get_stop_words("/home/kkmarmar/rag-search-engine/data/stopwords.txt")

    args = parser.parse_args()
    movies = load_mov("/home/kkmarmar/rag-search-engine/data/movies.json")
    results = []
    count = 0
    match args.command:
        case "search":
            print(f"Searching for {args.query}")
            args.query = tokenize(args.query)
            for movie in movies:
                title = tokenize(movie["title"])
                if token_match(args.query, title, stop_words, stemmer):
                    count += 1
                    results.append(f"{count}. {movie["title"]} {count}\n")
                if count >= 5:
                    break
            print(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()