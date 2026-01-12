from sklearn.feature_extraction.text import TfidfVectorizer

_vectorizer = None

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = TfidfVectorizer(stop_words="english")
    return _vectorizer
