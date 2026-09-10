from helper import *
import pickle
import os
import sys
import math
from collections import Counter
from collections import defaultdict

class InvertedIndex():
    
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_freq = defaultdict(Counter)

    def get_tf(self, doc_id, term):
        return self.term_freq[doc_id][term]
    
    def __add__document(self, doc_id, text):
        tokenized = tokenize(text)
        for token in set(tokenized):
            self.index[token].add(doc_id)
        self.term_freq[doc_id].update(tokenized)
                
    def get_documents(self, term):
        ids = self.index.get(term, set())
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
        with open("./cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_freq, file)

    def load(self):
        try:
            with open("./cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("./cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
            with open("./cache/term_frequencies.pkl", "rb") as file:
                self.term_freq = pickle.load(file)
        except Exception as e:
            raise Exception(e)

    def get_idf(self, term):
        doc_count = len(self.docmap)
        term_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_count + 1))

    def get_tf_idf(self, term, doc_id):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf
    
def single_token(term):
    res = tokenize(term)
    if len(res) != 1:
        raise Exception("Token is not a single token")
    return res

def tfidfer(term, doc_id):
    inverted = InvertedIndex()
    inverted.load()
    term = tokenize(term)
    return inverted.get_tf_idf(term[0], doc_id)

def idf_what(term):
    inverted = InvertedIndex()
    inverted.load()
    term = single_token(term)
    return inverted.get_idf(term[0])

def tf(doc_id, term):
    inverted = InvertedIndex()
    inverted.load()
    term = single_token(term)
    return inverted.get_tf(doc_id, term[0])

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