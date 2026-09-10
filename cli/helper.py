import string
import json

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
    res = []
    for word in str1:
        if word:
            res.append(word)
    return res

def token_match(str1: str, str2: str, stop_words: list[str], stemmer) -> bool:
    for word1 in str1:
        if word1 in stop_words:
            continue
        for word2 in str2:
            if word2 in stop_words:
                continue
            if stemmer.stem(word1) in stemmer.stem(word2):
                return True
    return False

def load_mov(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        movies = json.load(file)
        return movies["movies"]