import string
import json
from nltk.stem import PorterStemmer
def get_stop_words(filepath:str) -> list[str]:
    temp = None
    with open (filepath, "r", encoding="utf-8") as file:
        temp = file.read()
    temp = temp.split("\n")
    stop_words = []
    for word in temp:
        stop_words.append(process_text(word))
    return stop_words

def process_text(input: str) -> str:
    input = input.lower()
    input = input.translate(str.maketrans("", "", string.punctuation))
    return input

def tokenize(input: str) -> list[str]:
    str1 = process_text(input)
    str1 = str1.split()
    stemmer = PorterStemmer()
    stop_words = get_stop_words("/home/kkmarmar/rag-search-engine/data/stopwords.txt")
    res = []
    for word in str1:
        if word:
            if word in stop_words:
                continue
            word = stemmer.stem(word)
            res.append(word)
    return res

def load_mov(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        movies = json.load(file)
        return movies["movies"]