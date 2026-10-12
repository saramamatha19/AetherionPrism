"""Ask the LLM which sources to search (Phase 5).

Part 1: connect (make_client, check_connection).
Part 2: the routing prompt + structured output (ask_llm).
Part 3: never crash on temporary failures; a fixed backup plan (ask_llm_safely).
"""

import json
import time

import openai
from dotenv import load_dotenv
from openai import OpenAI

from prism.data import SOURCES

load_dotenv()  # reads OPENAI_API_KEY from the .env file (never from code or YAML)


# ---------- The instructions: copied from DATASET.md §4, so the LLM follows OUR labelling policy ----------

SOURCE_DESCRIPTIONS = """\
- kb: official help, how-to or policy articles
- jira: tickets, bugs, work status
- confluence: team docs: specs, designs, runbooks, process docs
- slack: chat conversations
- notion: notes: meeting notes, personal or team notes
- gmail: email
- web: public knowledge outside the company"""

LABELLING_RULES = """\
1. A named tool counts: "check Slack for…" includes slack.
2. Negation removes: "not in Slack" means slack is OUT.
3. Misleading words don't count: "parking ticket", "slack off", "the notion of" are not jira, slack or notion.
4. Specs, designs, runbooks, process docs → confluence. Meeting notes and personal notes → notion.
   Official policy or how-to → kb.
5. Public knowledge → web. Company topics are never web.
6. No information need (greetings, thanks, statements) → no sources (empty list).
7. Several sources only when clearly needed."""

SYSTEM_PROMPT_V1 = f"""You decide where to search to answer a workplace question.

Sources:
{SOURCE_DESCRIPTIONS}

Rules:
{LABELLING_RULES}

Answer with the list of sources to search (an empty list if no search is needed)."""


# Prompt v2 (Phase 5.7c): fixes from the failure report on the practice set (train, never eval):
# confluence was a default (added wrongly 39×), notion/slack missed, unclear questions spread over
# several sources, wrong second sources. Examples come from train but NOT from the practice set.
SYSTEM_PROMPT_V2 = """You decide where to search to answer a workplace question.

Sources:
- kb: official help, how-to or policy articles (customer-facing help, product how-tos, official policies)
- jira: tickets, bugs, tasks, work status (who owns it, is it fixed, blockers, progress)
- confluence: written team documents ONLY: specs, design docs, runbooks, process docs
- slack: chat conversations (what people said, suggested, recommended or discussed)
- notion: notes: meeting notes, decisions, plans, deadlines, personal or team notes
- gmail: email (what someone wrote, sent or replied)
- web: public knowledge outside the company

Rules:
1. A named tool counts: "check Slack for…" includes slack.
2. Negation removes: "not in Slack" means slack is OUT.
3. Misleading words don't count: "parking ticket", "slack off", "the notion of" are not jira, slack or notion.
4. Specs, designs, runbooks, process docs → confluence. Meeting notes, decisions and plans → notion.
   Official policy or how-to → kb.
5. Public knowledge → web. Company topics are never web.
6. No information need → NO sources (empty list). This includes greetings ("hi", "good morning"),
   thanks ("thanks", "that was helpful"), small talk ("how are you?", "tell me a joke"),
   questions about you, the assistant ("are you a bot?"), and requests to rephrase ("make it shorter").
7. Prefer ONE source. Add a second only if the question clearly asks for two different things.
8. Confluence is NEVER a default. Use it only when the answer is in a written spec, design, runbook
   or process doc. Status, owners and bugs → jira. Decisions and plans → notion. Discussions → slack.
9. If no tool is named, pick the single most likely place from the TYPE of information asked for.

Examples:
"tell me how you are doing" → []
"Can you make that a bit shorter?" → []
"Is there a jira for SAML onboarding with admin dry-run?" → ["jira"]
"Design doc for cooperative incremental rebalancing in the consumer client?" → ["confluence"]
"Any blockers reported for the server migration set for next week?" → ["jira"]
"help me make changes to my insurance policy" → ["kb"]
"How do cloud providers usually calculate service credits for SLA breaches?" → ["web"]
"I need the conversation with the invoice details from that European reseller." → ["gmail"]
"Not slack — did the customer write in about the latency spike after the v4beta rollout?" → ["gmail"]
"not chat, what's the official canary -> scale -> migrate checklist?" → ["kb"]
"how do I connect my existing business email to my website" → ["kb"]
"Does the timeline from the deployment channel match the one in the release plan doc?" → ["confluence", "slack"]
"Look into what we agreed on during the last meeting." → ["notion"]
"who owns the flaky tests now" → ["jira"]

Answer with the list of sources to search (an empty list if no search is needed)."""

# Prompt v3 (Phase 5.7, the LAST prompt edition): v2 + fixes from v2's failure report on the
# practice set: two sources for decisions / progress / plan-vs-doc (v2 gave one), deadlines → notion,
# "find / search" is never an empty answer, suggested fixes → slack. Same examples as v2 + 3 new
# ones (from train, not the practice set).
SYSTEM_PROMPT_V3 = """You decide where to search to answer a workplace question.

Sources:
- kb: official help, how-to or policy articles (customer-facing help, product how-tos, official policies)
- jira: tickets, bugs, tasks, work status (who owns it, is it fixed, blockers, progress)
- confluence: written team documents ONLY: specs, design docs, runbooks, process docs
- slack: chat conversations (what people said, suggested, recommended or discussed)
- notion: notes: meeting notes, decisions, plans, deadlines, personal or team notes
- gmail: email (what someone wrote, sent or replied)
- web: public knowledge outside the company

Rules:
1. A named tool counts: "check Slack for…" includes slack.
2. Negation removes: "not in Slack" means slack is OUT.
3. Misleading words don't count: "parking ticket", "slack off", "the notion of" are not jira, slack or notion.
4. Specs, designs, runbooks, process docs → confluence. Meeting notes, decisions and plans → notion.
   Official policy or how-to → kb.
5. Public knowledge → web. Company topics are never web.
6. No information need → NO sources (empty list). This includes greetings ("hi", "good morning"),
   thanks ("thanks", "that was helpful"), small talk ("how are you?", "tell me a joke"),
   questions about you, the assistant ("are you a bot?"), and requests to rephrase ("make it shorter").
   But if the user asks you to find, search, look up or check something, it IS an information need:
   never answer with an empty list.
7. Usually ONE source. Use TWO when the question asks for two different things, or in these
   common cases:
   - decisions or agreements ("did we agree / settle / decide…") → notion + slack
     (noted in meetings, discussed in chat). If the question mentions a meeting → notion only.
   - progress on a problem ("any progress on…", "did X get looked at", "what's happening with…")
     → jira + slack (the ticket and the discussion about it)
   - a plan or timeline checked against a document → confluence + jira
8. Confluence is NEVER a default. Use it only when the answer is in a written spec, design, runbook
   or process doc. Status, owners and bugs → jira. Decisions and plans → notion. Discussions → slack.
   Dates and deadlines ("when is X due", "what are the due dates", "target date") → notion,
   unless a ticket is named.
   A fix, mitigation or workaround that people suggested ("what change helped",
   "what mitigation was suggested") → slack.
9. If no tool is named, pick the single most likely place from the TYPE of information asked for.

Examples:
"tell me how you are doing" → []
"Can you make that a bit shorter?" → []
"Is there a jira for SAML onboarding with admin dry-run?" → ["jira"]
"Design doc for cooperative incremental rebalancing in the consumer client?" → ["confluence"]
"Any blockers reported for the server migration set for next week?" → ["jira"]
"help me make changes to my insurance policy" → ["kb"]
"How do cloud providers usually calculate service credits for SLA breaches?" → ["web"]
"I need the conversation with the invoice details from that European reseller." → ["gmail"]
"Not slack — did the customer write in about the latency spike after the v4beta rollout?" → ["gmail"]
"not chat, what's the official canary -> scale -> migrate checklist?" → ["kb"]
"how do I connect my existing business email to my website" → ["kb"]
"Does the timeline from the deployment channel match the one in the release plan doc?" → ["confluence", "slack"]
"Look into what we agreed on during the last meeting." → ["notion"]
"who owns the flaky tests now" → ["jira"]
"did we agree a policy on third party libraries" → ["notion", "slack"]
"any progress on the memory usage problem" → ["jira", "slack"]
"Are the deadlines in the epic consistent with the timeline proposed in the roadmap plan?" → ["confluence", "jira"]

Answer with the list of sources to search (an empty list if no search is needed)."""

# Prompt v4 (Phase 5.7, the FINAL edition): v3 + a narrower rule 7. On eval, v3 added slack as an
# extra source on many single-source questions (e.g. "has the export timeout been closed" → jira,
# slack). v4 says when two sources are right and, explicitly, when NOT to add slack. 3 more examples
# (from train, not the practice set, never eval).
SYSTEM_PROMPT_V4 = """You decide where to search to answer a workplace question.

Sources:
- kb: official help, how-to or policy articles (customer-facing help, product how-tos, official policies)
- jira: tickets, bugs, tasks, work status (who owns it, is it fixed, blockers, progress)
- confluence: written team documents ONLY: specs, design docs, runbooks, process docs
- slack: chat conversations (what people said, suggested, recommended or discussed)
- notion: notes: meeting notes, decisions, plans, deadlines, personal or team notes
- gmail: email (what someone wrote, sent or replied)
- web: public knowledge outside the company

Rules:
1. A named tool counts: "check Slack for…" includes slack.
2. Negation removes: "not in Slack" means slack is OUT.
3. Misleading words don't count: "parking ticket", "slack off", "the notion of" are not jira, slack or notion.
4. Specs, designs, runbooks, process docs → confluence. Meeting notes, decisions and plans → notion.
   Official policy or how-to → kb.
5. Public knowledge → web. Company topics are never web.
6. No information need → NO sources (empty list). This includes greetings ("hi", "good morning"),
   thanks ("thanks", "that was helpful"), small talk ("how are you?", "tell me a joke"),
   questions about you, the assistant ("are you a bot?"), and requests to rephrase ("make it shorter").
   But if the user asks you to find, search, look up or check something, it IS an information need:
   never answer with an empty list.
7. Usually ONE source. Use TWO only when the question asks for two different things, or in
   these three cases:
   - "did we agree / did we settle on / where did we get to on X" (was a decision reached?)
     → notion + slack (noted in meetings, discussed in chat)
   - an open request for updates on a problem ("any progress on…", "any updates on…",
     "what's happening with…", "did X get looked at") → jira + slack
   - a plan or timeline checked against a document → confluence + jira
   Do NOT add slack just because people might have talked about it. In particular:
   - a yes/no status question about one item ("has X been fixed / closed / merged / triaged",
     "is the PR merged", "did they trim that") → jira only
   - "what did we decide / what was the decision on X" → notion only
   - what a group or team said or discussed in a meeting, or how they came up with something
     → notion only (meeting notes). If the question mentions a meeting → notion only.
8. Confluence is NEVER a default. Use it only when the answer is in a written spec, design, runbook
   or process doc. Status, owners and bugs → jira. Decisions and plans → notion. Discussions → slack.
   Dates and deadlines ("when is X due", "what are the due dates", "target date") → notion,
   unless a ticket is named.
   A fix, mitigation or workaround that people suggested ("what change helped",
   "what mitigation was suggested") → slack.
9. If no tool is named, pick the single most likely place from the TYPE of information asked for.

Examples:
"tell me how you are doing" → []
"Can you make that a bit shorter?" → []
"Is there a jira for SAML onboarding with admin dry-run?" → ["jira"]
"Design doc for cooperative incremental rebalancing in the consumer client?" → ["confluence"]
"Any blockers reported for the server migration set for next week?" → ["jira"]
"help me make changes to my insurance policy" → ["kb"]
"How do cloud providers usually calculate service credits for SLA breaches?" → ["web"]
"I need the conversation with the invoice details from that European reseller." → ["gmail"]
"Not slack — did the customer write in about the latency spike after the v4beta rollout?" → ["gmail"]
"not chat, what's the official canary -> scale -> migrate checklist?" → ["kb"]
"how do I connect my existing business email to my website" → ["kb"]
"Does the timeline from the deployment channel match the one in the release plan doc?" → ["confluence", "slack"]
"Look into what we agreed on during the last meeting." → ["notion"]
"who owns the flaky tests now" → ["jira"]
"did we agree a policy on third party libraries" → ["notion", "slack"]
"any progress on the memory usage problem" → ["jira", "slack"]
"Is the PR merged that reduces GPU peak memory with arena tuning and checkpointing?" → ["jira"]
"what did we decide for billing-rate guardrails in the tiered throttle rollout proposal" → ["notion"]
"What did the group discuss about remote control design?" → ["notion"]
"Are the deadlines in the epic consistent with the timeline proposed in the roadmap plan?" → ["confluence", "jira"]

Answer with the list of sources to search (an empty list if no search is needed)."""

# Every prompt version ever used. NEVER edit an existing one: add a new version instead,
# so old recordings (whose file names include the prompt's fingerprint) stay meaningful.
PROMPTS = {
    "v1": SYSTEM_PROMPT_V1,
    "v2": SYSTEM_PROMPT_V2,
    "v3": SYSTEM_PROMPT_V3,
    "v4": SYSTEM_PROMPT_V4,
}


def system_prompt(settings) -> str:
    """The prompt chosen in YAML (llm_prompt_version). Unknown version → error."""
    if settings.llm_prompt_version not in PROMPTS:
        raise ValueError(
            f"Unknown llm_prompt_version '{settings.llm_prompt_version}'. Known: {list(PROMPTS)}"
        )
    return PROMPTS[settings.llm_prompt_version]


# ---------- Structured output: the reply MUST be {"sources": [...]} with only our 7 labels ----------

ANSWER_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "sources_to_search",
        "strict": True,  # the API guarantees the reply follows this shape exactly
        "schema": {
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",  # several, one, or none (empty = no search)
                    "items": {"type": "string", "enum": list(SOURCES)},  # only our 7 labels
                }
            },
            "required": ["sources"],
            "additionalProperties": False,  # no extra fields, no chatty explanations
        },
    },
}


class RefusalError(Exception):
    """The LLM declined to answer."""


def make_client(settings) -> OpenAI:
    """An OpenAI client with our timeout and retry settings. It finds the key in .env by itself."""
    return OpenAI(timeout=settings.llm_timeout_seconds, max_retries=settings.llm_max_retries)


def check_connection():
    """Send one tiny request to prove the key and the model name work."""
    from prism.config import load_settings

    settings = load_settings()
    client = make_client(settings)
    reply = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": "Reply with the single word OK."}],
    )
    print(f"Model {settings.llm_model} replied: {reply.choices[0].message.content}")
    print(f"Tokens used: {reply.usage.prompt_tokens} in, {reply.usage.completion_tokens} out")


def ask_llm(client, settings, question_text: str) -> dict:
    """Ask the LLM where to search. Returns the sources + the real token counts + how long it took."""
    extra_settings = {}
    if not settings.llm_model.startswith("gpt-5"):  # reasoning models (gpt-5…) don't accept it
        extra_settings["temperature"] = 0  # no creativity: the same question → the same answer

    start = time.perf_counter()
    reply = client.chat.completions.create(
        model=settings.llm_model,
        messages=[
            {"role": "system", "content": system_prompt(settings)},  # the instructions
            {"role": "user", "content": question_text},  # the question
        ],
        response_format=ANSWER_FORMAT,
        **extra_settings,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    message = reply.choices[0].message
    if message.refusal:  # the LLM said "I can't help with that"
        raise RefusalError(message.refusal)

    answer = json.loads(message.content)  # '{"sources": ["slack"]}' → a dict
    for source in answer["sources"]:  # extra safety: every label must be one of our 7
        if source not in SOURCES:
            raise ValueError(f"Unknown source from the LLM: {source}")

    # Real token counts from the API's own billing numbers (never estimated: prototype mistake #8)
    usage = reply.usage
    cached_tokens = 0
    if usage.prompt_tokens_details is not None:
        cached_tokens = usage.prompt_tokens_details.cached_tokens  # cheaper repeated part (#7)

    return {
        "sources": frozenset(answer["sources"]),
        "prompt_tokens": usage.prompt_tokens,
        "cached_tokens": cached_tokens,
        "completion_tokens": usage.completion_tokens,
        "latency_ms": latency_ms,
    }


def ask_llm_safely(client, settings, question_text: str, model_answer=None) -> dict:
    """Ask the LLM, but NEVER crash on a temporary or one-off failure.

    Returns what the user should get (`sources`), what the LLM said (`llm_sources`, or None),
    what went wrong (`llm_error`, or None) and which backup plan was used (`fallback`, or None).
    Setup problems (wrong key, no permission, bad request = our bug) are NOT hidden: they stop loudly.
    The OpenAI client has already retried temporary errors `llm_max_retries` times before we get here.
    """
    try:
        result = ask_llm(client, settings, question_text)
        result["llm_sources"] = result["sources"]
        result["llm_error"] = None
        result["fallback"] = None
        return result

    except (openai.AuthenticationError, openai.PermissionDeniedError, openai.BadRequestError):
        raise  # our setup is broken → stop loudly and fix it

    # Temporary or one-off problems → name them in plain words, then use the backup plan
    except (
        openai.APITimeoutError
    ):  # must come before APIConnectionError (a timeout is a kind of it)
        error = "timeout"
    except openai.RateLimitError:
        error = "rate limit"
    except openai.APIConnectionError:
        error = "connection error"
    except openai.APIStatusError as e:
        error = f"server error {e.status_code}"
    except RefusalError:
        error = "refusal"
    except (json.JSONDecodeError, KeyError, ValueError):
        error = "bad reply"

    # The backup plan: the model's answer if it has one, otherwise search everything
    # (the question came here because the model was unsure, so an empty model answer isn't trusted)
    if model_answer:
        sources = frozenset(model_answer)
        fallback = "model answer"
    else:
        sources = frozenset(SOURCES)
        fallback = "all sources"

    return {
        "sources": sources,
        "llm_sources": None,
        "llm_error": error,
        "fallback": fallback,
        "prompt_tokens": 0,
        "cached_tokens": 0,
        "completion_tokens": 0,
        "latency_ms": None,
    }
