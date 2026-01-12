import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

VECTOR_STORE_PATH = "backend/data/vector_store"

def retrieve(query: str, top_k: int = 3):
    with open(f"{VECTOR_STORE_PATH}/tfidf.pkl", "rb") as f:
        store = pickle.load(f)

    vectorizer = store["vectorizer"]
    embeddings = store["embeddings"]
    texts = store["texts"]
    sources = store["sources"]

    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, embeddings)[0]

    top_indices = similarities.argsort()[-top_k:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            "source": sources[idx],
            "content": texts[idx],
            "score": float(similarities[idx])
        })

    return results
