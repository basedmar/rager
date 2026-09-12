from sentence_transformers import SentenceTransformer
import numpy as np
import os
import json
from numpy.typing import NDArray
from typing import Any
EmbeddingArray = NDArray[Any]

class SemanticSearch():
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)        
        self.embeddings: EmbeddingArray = None
        self.documents: list[Any] | None = None
        self.document_map: dict[int, Any] = {}

    def build_embedding(self, documents):
        self.documents = documents
        mov_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            mov_strings.append(f"{doc["title"]}: {doc["description"]}")
        self.embeddings = self.model.encode(mov_strings, show_progress_bar=True)
        os.makedirs("./cache", exist_ok=True)
        with open("./cache/movie_embeddings.npy", "wb") as file:
            np.save(file, self.embeddings)
        return self.embeddings

    def load_or_make_embed(self, documents):
        self.documents = documents
        for doc in documents:
            self.document_map[doc["id"]] = doc
        if os.path.exists("./cache/movie_embeddings.npy"):
            with open("./cache/movie_embeddings.npy", "rb") as file:
                self.embeddings = np.load(file)
            if self.embeddings.shape[0] == len(documents):
                return self.embeddings
        return self.build_embedding(documents)
    
    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("provided text is empty or is only whitespace")
        return self.model.encode([text])[0]

    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        if self.documents is None or len(self.documents) == 0:
            raise ValueError("No documents loaded. Call `load_or_create_embeddings` first.")
        embed_query = self.generate_embedding(query)
        search_results = []
        for index, movie in enumerate(self.embeddings):
            search_results.append((cosine_similarity(embed_query, movie), self.document_map[index+1]))
        search_results = sorted(search_results, key=lambda x: x[0], reverse=True)
        res = []
        for i in range(limit):
            res.append({"score": search_results[i][0], "title": search_results[i][1]["title"], "description": search_results[i][1]["description"]})
        return res

def searcher(query, limit):
    sem = SemanticSearch()
    documents = load_mov("./data/movies.json")
    sem.load_or_make_embed(documents)
    return sem.search(query, limit)

def cosine_similarity(vec1, vec2):
    dot_val = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return dot_val / (norm1 * norm2)


def embed_user_query(query):
    sem = SemanticSearch()
    user_vec = sem.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {user_vec[:3]}")
    print(f"Shape: {user_vec.shape}")

def load_mov(filepath: str):
    with open(filepath, "r", encoding="utf-8") as file:
        movies = json.load(file)
        return movies["movies"]
    
def verify_embedding():
    sem = SemanticSearch()
    documents = load_mov("./data/movies.json")
    embeddings = sem.load_or_make_embed(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
    f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
def embed_text(text):
    model = SemanticSearch()
    embed = model.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embed[:3]}")
    print(f"Dimensions: {embed.shape[0]}")

def verify_model():
    model = SemanticSearch()
    print(f"Model Loaded: {model.model}")
    print(f"Max sequence length: {model.model.max_seq_length}")