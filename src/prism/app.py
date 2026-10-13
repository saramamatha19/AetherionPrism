"""The Aetherion Prism page (Streamlit): a thin WINDOW over router.route().

It decides nothing: it sends the question to the router and shows the DecisionRecord.
All the logic lives in router.py (so if something looks wrong, debug the router, not the page).

Run:  uv run streamlit run src/prism/app.py
"""

import streamlit as st

from prism.decisionLog import DecisionLog, count_shadow_questions
from prism.router import Router

MODES = {
    "Shadow (trial): the LLM answers, the model is recorded": "shadow",
    "Gated: the model when sure, the LLM otherwise": "gated",
    "Model only: no LLM (testing)": "model_only",
}


@st.cache_resource  # made ONCE when the app starts, then shared by every user (~36 s to load)
def get_router() -> Router:
    return Router(log=DecisionLog())


st.title("Aetherion Prism")
st.caption("Where should I search? A small model answers when it's sure; an LLM answers the rest.")

router = get_router()

# Remember this tab's numbers between clicks (Streamlit re-runs the script on every click)
if "questions" not in st.session_state:
    st.session_state["questions"] = 0
    st.session_state["llm_calls"] = 0
    st.session_state["cost"] = 0.0

mode_label = st.radio("Mode", list(MODES.keys()), index=0)  # index 0 = Shadow, the default
st.caption("Shadow is the trial mode: please keep it unless you're testing.")

question = st.text_area("Your question", placeholder="e.g. is the login bug fixed?")


def sources_line(sources) -> str:
    """{"slack", "jira"} → "jira, slack"; empty → "No search needed"; None → "—" (not asked).
    Formatting only: the page decides nothing."""
    if sources is None:
        return "—"
    if not sources:
        return "No search needed"
    return ", ".join(sorted(sources))


if st.button("Route") and question.strip():
    record = router.route(question.strip(), mode=MODES[mode_label])
    st.session_state["questions"] += 1
    if record.llm_answer is not None or record.llm_error is not None:  # the LLM was asked
        st.session_state["llm_calls"] += 1
    st.session_state["cost"] += record.cost_spent

    # 1. The answer the user gets
    st.subheader(f"Search in: {sources_line(record.answer)}")
    st.write(f"Decided by: **{record.decided_by.upper()}** — {record.reason}")

    # 2. Four numbers side by side (all straight from the record)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Time", f"{record.latency_ms:,.0f} ms")
    col2.metric("Cost", f"${record.cost_spent:.5f}")
    col3.metric("Model confidence", f"{record.model_confidence:.2f}", help=f"gate {record.gate}")
    if record.would_escalate:
        col4.metric("Gated mode would…", "ask the LLM")
    else:
        col4.metric("Gated mode would…", "let the model answer")

    # 3. What each one said
    if record.decided_by == "model":
        st.write(f"**Model said:** {sources_line(record.model_answer)}")
    else:
        st.write(f"**Model said (recorded only):** {sources_line(record.model_answer)}")
    st.write(f"**LLM said:** {sources_line(record.llm_answer)}")
    if record.llm_error:
        st.warning(f"The LLM failed ({record.llm_error}); the backup answer was used.")


# The sidebar (drawn last so the session numbers include the question just asked)
with st.sidebar:
    st.header("Setup")
    st.write(f"Default mode: **{router.mode}**")
    st.write(f"Model: `{router.model_version}`")
    st.write(f"LLM: `{router.settings.llm_model}` · prompt {router.settings.llm_prompt_version}")
    st.write(f"Gate: {router.gate}")

    st.header("This session")
    st.caption("Your tab only")
    st.write(f"Questions: {st.session_state['questions']}")
    st.write(f"LLM calls: {st.session_state['llm_calls']}")
    st.write(f"Cost: ${st.session_state['cost']:.4f}")

    st.header("Shadow trial")
    goal = router.settings.go_live_min_questions  # 500, from the YAML
    done = count_shadow_questions()
    st.write(f"{done} / {goal} shadow questions")
    st.progress(min(done / goal, 1.0))
