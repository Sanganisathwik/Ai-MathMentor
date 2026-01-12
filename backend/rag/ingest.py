import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from backend.rag.embeddings import get_vectorizer

KB_PATH = "backend/rag/knowledge_base"
VECTOR_STORE_PATH = "backend/data/vector_store"

def ingest():
    texts = []
    sources = []

    for file in os.listdir(KB_PATH):
        if file.endswith(".md"):
            with open(os.path.join(KB_PATH, file), "r", encoding="utf-8") as f:
                content = f.read()
                chunks = content.split("\n\n")
                for chunk in chunks:
                    if len(chunk.strip()) > 20:
                        texts.append(chunk)
                        sources.append(file)

    vectorizer = get_vectorizer()
    embeddings = vectorizer.fit_transform(texts)

    os.makedirs(VECTOR_STORE_PATH, exist_ok=True)

    with open(os.path.join(VECTOR_STORE_PATH, "tfidf.pkl"), "wb") as f:
        pickle.dump(
            {
                "vectorizer": vectorizer,
                "embeddings": embeddings,
                "texts": texts,
                "sources": sources
            },
            f
        )

    print("✅ Knowledge base ingested successfully (TF-IDF)")
