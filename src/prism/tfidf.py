"""Turn questions into numbers with TF-IDF: whole words + 3–5 letter chunks."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf() -> FeatureUnion:
    """Two TF-IDFs side by side: one on words, one on letter chunks."""
    words = TfidfVectorizer(ngram_range=(1, 2))
    chunks = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5))
    return FeatureUnion([("words", words), ("chunks", chunks)])
