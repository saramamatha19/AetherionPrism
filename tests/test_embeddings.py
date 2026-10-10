"""Tests for the sentence-embedding adapter (uses BGE; takes a few seconds to load)."""

import joblib

from prism.embeddings import SentenceEmbeddings

BGE = "BAAI/bge-small-en-v1.5"


def test_each_question_becomes_384_numbers():
    numbers = SentenceEmbeddings(BGE).fit_transform(["is the bug fixed?", "thanks!"])
    assert numbers.shape == (2, 384)


def test_similar_meaning_is_closer():
    v = SentenceEmbeddings(BGE).fit_transform(
        ["is the bug fixed?", "has the defect been resolved?", "what is a KV cache?"]
    )
    assert v[0] @ v[1] > v[0] @ v[2]  # bug ≈ defect, closer than an unrelated question


def test_saved_file_is_small_and_still_works(tmp_path):
    emb = SentenceEmbeddings(BGE).fit(["x"])
    before = emb.transform(["is the bug fixed?"])
    joblib.dump(emb, tmp_path / "emb.joblib")
    assert (tmp_path / "emb.joblib").stat().st_size < 10_000  # only the name, not 130 MB
    loaded = joblib.load(tmp_path / "emb.joblib")
    assert (loaded.transform(["is the bug fixed?"]) == before).all()
