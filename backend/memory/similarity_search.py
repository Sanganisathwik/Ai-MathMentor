from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def find_similar(memory_records, query, threshold=0.6):
    if not memory_records:
        return []

    texts = [m["input_text"] for m in memory_records]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(texts + [query])

    similarities = cosine_similarity(vectors[-1], vectors[:-1])[0]

    similar = []
    for i, score in enumerate(similarities):
        if score >= threshold:
            similar.append((memory_records[i], float(score)))

    return similar
