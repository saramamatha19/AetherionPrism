"""Turn questions into numbers with a sentence-embedding model (meaning, not words).

Works with any model on Hugging Face (BGE, MiniLM, E5, …): only the name changes.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.base import BaseEstimator, TransformerMixin


class SentenceEmbeddings(BaseEstimator, TransformerMixin):
    """Any sentence-embedding model with the same buttons as TF-IDF, so it plugs into logreg.py."""

    def __init__(self, model_name: str):
        self.model_name = model_name  # e.g. "BAAI/bge-small-en-v1.5"; no default: always say which
        self.encoder = None  # loaded when first needed

    def load(self):
        """Load the model (from the laptop's cache after the first download)."""
        if self.encoder is None:
            self.encoder = SentenceTransformer(self.model_name)

    def fit(self, texts, y=None):
        """The model is already trained, so there's nothing to learn. Just make sure it's loaded."""
        self.load()
        return self

    def transform(self, texts):
        """Each question → its numbers (384 for small models, 768 for base models)."""
        self.load()
        return self.encoder.encode(list(texts), normalize_embeddings=True)

    def get_feature_names_out(self, input_features=None):
        """Names for the columns: emb_0, emb_1, … (needed by FeatureUnion and top_words)."""
        self.load()
        names = []
        for i in range(self.encoder.get_sentence_embedding_dimension()):
            names.append(f"emb_{i}")
        return np.array(names)

    def __getstate__(self):
        """When the model is saved: keep only the model's NAME, not the model itself (~130 MB)."""
        state = self.__dict__.copy()
        state["encoder"] = None
        return state
