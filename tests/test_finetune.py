"""Tests for making fine-tuning pairs."""

from prism.data import Question
import pytest

from prism.finetune import NEGATIONS, augment_negations, finetune, make_pairs, overlap


def q(id, text, *labels):
    return Question(id=id, text=text, labels=frozenset(labels), question_type="", part="")


def test_overlap_by_hand():
    assert overlap(frozenset({"slack"}), frozenset({"slack"})) == 1.0
    assert overlap(frozenset({"jira"}), frozenset({"jira", "confluence"})) == 0.5
    assert overlap(frozenset({"slack"}), frozenset({"gmail"})) == 0.0
    assert overlap(frozenset(), frozenset()) == 1.0


TINY = [
    q("1", "not the email, what did people say in slack", "slack"),
    q("2", "what did the team say in the channel", "slack"),
    q("3", "did the client reply to my email", "gmail"),
    q("4", "check my inbox for the invoice", "gmail"),
]


def test_tricky_pair_for_negation():
    pairs, counts = make_pairs(TINY, seed=42)
    assert counts["tricky"] >= 1  # question 1 mentions "email" but doesn't need gmail
    tricky = []
    for a, b, target in pairs:
        if a == TINY[0].text and target == 0.0:
            tricky.append(b)
    assert TINY[2].text in tricky or TINY[3].text in tricky  # paired with a gmail question


def test_same_seed_gives_same_pairs():
    assert make_pairs(TINY, seed=42) == make_pairs(TINY, seed=42)


def test_finetune_never_overwrites(tmp_path):
    with pytest.raises(ValueError, match="never overwritten"):
        finetune("any-model", [], 1, 16, 42, out_dir=tmp_path)  # tmp_path already exists


def test_augmented_questions_keep_their_labels():
    augmented = augment_negations(TINY, count=10, seed=42)
    assert len(augmented) == 10
    for new_text, original_text, labels, excluded in augmented:
        assert excluded not in labels  # only negate a source the question doesn't need
        assert new_text.endswith(original_text[1:])  # the original question is kept
        starts_with_phrase = False
        for phrase in NEGATIONS[excluded]:
            if new_text.startswith(phrase + ", "):
                starts_with_phrase = True
        assert starts_with_phrase
