"""Simple rules that send known weak spots to the LLM, whatever the model's confidence.

Why: the model is confidently wrong on many negation questions ("not the email, …"),
and the gate can't catch them because the model doesn't know it's unsure. The LLM
understands "not", so these questions go straight to it (Phase 4c).
"""
# for tfidf_ftv2_bge_loreg model as it covers negation hard case question from answering wrong with confidence

import re

# Words that usually mean "don't search X" (labelling rule 2: negation removes)
NEGATION_WORDS = [
    "not",
    "don't",
    "dont",
    "do not",
    "except",
    "skip",
    "instead of",
    "rather than",
    "other than",
    "without",
    "forget",
    "ignore",
]

# Whole words only (\b), so "not" matches but "notion" and "notes" don't
NEGATION_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(word) for word in NEGATION_WORDS) + r")\b"
)


def has_negation(text: str) -> bool:
    """True if the question contains a negation word → the model's weak spot → send to the LLM."""
    return NEGATION_PATTERN.search(text.lower()) is not None
