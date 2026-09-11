from helper import *
import pickle
import os
import sys
import math
from collections import Counter
from collections import defaultdict
from constants import *
class InvertedIndex():
    
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_freq = defaultdict(Counter)
        self.doc_lengths = {}

    def get_tf(self, doc_id, term):
        return self.term_freq[doc_id][term]
    
    def __add__document(self, doc_id, text):
        tokenized = tokenize(text)
        for token in set(tokenized):
            self.index[token].add(doc_id)
        self.term_freq[doc_id].update(tokenized)
        count = len(tokenized)
        self.doc_lengths[doc_id] = count

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
        with open("./cache/doclengths.pkl", "wb") as file:
            pickle.dump(self.doc_lengths, file)
        with open("./cache/index.pkl", "wb") as file:
            pickle.dump(self.index, file)
        with open("./cache/docmap.pkl", "wb") as file:
            pickle.dump(self.docmap, file)
        with open("./cache/term_frequencies.pkl", "wb") as file:
            pickle.dump(self.term_freq, file)

    def load(self):
        try:
            with open("./cache/doclengths.pkl", "rb") as file:
                self.doc_lengths = pickle.load(file)
            with open("./cache/index.pkl", "rb") as file:
                self.index = pickle.load(file)
            with open("./cache/docmap.pkl", "rb") as file:
                self.docmap = pickle.load(file)
            with open("./cache/term_frequencies.pkl", "rb") as file:
                self.term_freq = pickle.load(file)
        except Exception as e:
            raise Exception(e)

    def __get_avg_doc_len(self):
        total = 0
        for doc in self.doc_lengths:
            total += self.doc_lengths[doc]
        return total / len(self.doc_lengths)

    def get_idf(self, term):
        doc_count = len(self.docmap)
        term_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_count + 1))

    def get_tf_idf(self, term, doc_id):
        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        return tf * idf

    def get_bm25_idf(self, term):
        doc_count = len(self.docmap)
        term_count = len(self.index[term])
        return math.log((doc_count - term_count + 0.25) / (term_count + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        length_norm = 1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_len())
        raw_freq = self.get_tf(doc_id, term)
        return (raw_freq * (k1 + 1) / (raw_freq + k1 * length_norm))

    def bm25(self, doc_id, term):
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
 
    def bm25search(self, query, limit=5):
        tokens = tokenize(query)
        results = []
        for doc in self.docmap:
            sum = 0
            for token in tokens:
                if doc not in self.index[token]:
                    continue
                sum += self.bm25(doc, token)
            results.append((doc, sum))
        res = sorted(results, key=lambda x: x[1], reverse=True)
        return res[0:limit]

def get_bm25_tf(doc_id, term, k1, b):
    inverted = InvertedIndex()
    inverted.load()
    term = single_token(term)
    return inverted.get_bm25_tf(doc_id, term[0], k1, b)

def get_bm25_idf(term):
    inverted = InvertedIndex()
    inverted.load()
    term = single_token(term)
    return inverted.get_bm25_idf(term[0])

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