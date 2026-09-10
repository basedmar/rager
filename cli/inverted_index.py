from helper import *
import pickle
import os
import sys
from nltk.stem import PorterStemmer
class InvertedIndex():
    def __init__(self):
        self.index = {}
        self.docmap = {}

    def __add__document(self, doc_id, text):
        stemmer = PorterStemmer()
        stop_words = get_stop_words("/home/kkmarmar/rag-search-engine/data/stopwords.txt")
        tokenized = tokenize(text, stop_words, stemmer)
        for token in set(tokenized):
            if token in self.index:
                self.index[token].append(doc_id)
            else:
                self.index[token] = [doc_id]

    def get_documents(self, term):
        ids = set()
        if term not in self.index:
            return []
        for doc_id in self.index[term]:
            ids.add(doc_id)
        return sorted(list(ids))

    def build(self):
        movies = load_mov("/home/kkmarmar/rag-search-engine/data/movies.json")
        for movie in movies:
            text = f"{movie["title"]} {movie["description"]}"
            self.__add__document(movie["id"], text)
            self.docmap[movie["id"]] = movie
    
    def save(self):
        os.makedirs("./cache", exist_ok=True)
        with open("./cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
        with open("./cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)

    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("./cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
        except Exception as e:
            raise Exception(e)

def build():
    inverted = InvertedIndex()
    inverted.build()
    inverted.save()

def search(query) -> list[dict]:
    inv_index = InvertedIndex()
    try:
        inv_index.load()
    except Exception as e:
        print(e)
        sys.exit(1)
    tokenized = tokenize(query)
    passed, res = set(), []
    for token in tokenized:
        ids = inv_index.get_documents(token)
        for id in ids:
            if id in passed:
                continue
            passed.add(id)
            res.append(inv_index.docmap[id])
            if len(res) >= 5:
                return res
    return res