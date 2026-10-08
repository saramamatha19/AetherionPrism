"""Tests for TF-IDF: rare words count more than common ones."""

from prism.tfidf import build_tfidf

SENTENCES = ["is the bug fixed", "is the email sent", "the bug in the email"]


def test_rare_word_counts_more_than_common_word():
    tfidf = build_tfidf()
    numbers = tfidf.fit_transform(SENTENCES)
    columns = list(tfidf.get_feature_names_out())

    q1 = numbers[0].toarray()[0]  # the numbers for "is the bug fixed"
    fixed = q1[columns.index("words__fixed")]  # in 1 of 3 sentences → rare
    the = q1[columns.index("words__the")]  # in 3 of 3 sentences → common
    assert fixed > the
