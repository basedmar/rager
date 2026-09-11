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
        case "bm25idf":
            output = get_bm25_idf(args.term)
            print(f"BM25 IDF score of '{args.term}': {output:.2f}")
        case "bm25tf":
            doc_id = args.doc_id
            term = args.term
            k1 = args.k1
            b = args.b
            bm25tf = get_bm25_tf(doc_id, term, k1, b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")

        case "bm25search":
            inverted = InvertedIndex()
            inverted.load()
            term = args.term
            result = inverted.bm25search(term)
            for res in result:
                print(f"({res[0]}) {inverted.docmap[res[0]]["title"]} - Score: {res[1]:.2f}")
                
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()