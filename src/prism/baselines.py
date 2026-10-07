"""Baseline "student" to beat: clue words only, no learning. Deliberately simple."""

import re

# Clue words per source, written from the labelling rules (DATASET.md §4), not from eval.
KEYWORDS = {
    "jira": [
        "jira", "ticket", "tickets", "bug", "bugs", "issue", "issues", "sprint",
        "epic", "backlog", "assignee", "assigned", "blocker", "regression",
        "fixed", "released", "resolved", "priority",
    ],
    "slack": [
        "slack", "channel", "chat", "dm", "thread", "message", "messages",
        "pinged", "huddle", "posted", "conversation", "discussion",
    ],
    "gmail": [
        "email", "emails", "e-mail", "mail", "gmail", "inbox", "reply",
        "replied", "attachment", "cc", "newsletter",
    ],
    "confluence": [
        "confluence", "spec", "specs", "specification", "design doc", "design document",
        "runbook", "playbook", "architecture", "wiki", "process doc", "documentation",
    ],
    "notion": [
        "notion", "notes", "meeting notes", "minutes", "meeting", "agenda",
        "standup", "retro", "decided", "to-do", "todo",
    ],
    "kb": [
        "how do i", "how to", "how can i", "policy", "policies", "set up", "setup",
        "reset", "password", "benefits", "payroll", "expense", "faq", "help article",
    ],
    "web": [
        "what is", "what are", "who is", "who was", "who played", "define",
        "definition", "meaning of", "capital of", "history of", "wikipedia", "google",
    ],
}  # fmt: skip

# One search pattern per source. \b = whole words only, so "dm" doesn't match "admin".
PATTERNS = {
    source: re.compile(r"\b(" + "|".join(re.escape(word) for word in words) + r")\b")
    for source, words in KEYWORDS.items()
}


def keyword_rules(text: str) -> frozenset[str]:
    """Pick every source whose clue words appear in the question. No clue → no search."""
    text = text.lower()
    return frozenset(source for source, pattern in PATTERNS.items() if pattern.search(text))
