import argparse
import json
import string
from helper import *
from nltk.stem import PorterStemmer
def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    stemmer = PorterStemmer()
    search_parser = subparsers.add_parser("search", help="Search movies using keywords")
    search_parser.add_argument("query", type=str, help="Search query")
    temp = None
    with open("/home/kkmarmar/rag-search-engine/data/stopwords.txt") as file:
        temp = file.read()
    temp = temp.split("\n")
    stop_words = []
    for word in temp:
        stop_words.append(process_text(word))

    args = parser.parse_args()
    movies = None
    results = []
    count = 0
    match args.command:
        case "search":
            print(f"Searching for {args.query}")
            with open("/home/kkmarmar/rag-search-engine/data/movies.json", "r", encoding="utf-8") as file:
                movies = json.load(file)
            args.query = tokenize(args.query)
            for movie in movies["movies"]:
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