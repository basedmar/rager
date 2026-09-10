from parser import *
from helper import *
from inverted_index import *
import sys

def main() -> None:
    match args.command:
        case "search":
            print(f"Searching for {args.query}")
            results = search(args.query)
            for count, res in enumerate(results):
                print(f"{count}. {res} {count}")
        case "build":
            build()
        case "tf":
            doc_id = args.doc_id
            term = args.term
            print(tf(doc_id, term))
        case "idf":
            term = args.term
            idf = idf_what(term)
            print(f"Inverse document frequency of `{args.term}`: {idf:.2f}")
        case "tfidf":
            term = args.term
            doc_id = args.doc_id
            tf_idf = tfidfer(term, doc_id)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()