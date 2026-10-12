"""A FAKE LLM for tests: replies or fails on purpose (free, instant, no real API call).

Used by test_llm.py and test_llm_answer_eval.py.
"""

from types import SimpleNamespace

import httpx2

REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def fake_reply(content, refusal=None):
    """Something shaped like a real OpenAI reply."""
    message = SimpleNamespace(content=content, refusal=refusal)
    usage = SimpleNamespace(prompt_tokens=300, completion_tokens=5, prompt_tokens_details=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


class FakeClient:
    """Pretends to be the OpenAI client: returns a chosen reply, or raises a chosen error."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        if self.error is not None:
            raise self.error
        return self.reply
