"""The Aetherion Prism page (Streamlit): a thin WINDOW over router.route().

It decides nothing: it sends the question to the router and shows the DecisionRecord
(and the record's own helpers: money_saved, model_and_llm_agree). All the logic lives in
router.py / decisionRecord.py, so if something looks wrong, debug those, not the page.

Run:  uv run streamlit run src/prism/app.py          (the look: .streamlit/config.toml)
"""

import html
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from prism.decisionLog import DecisionLog, count_shadow_questions
from prism.registry import MODELS_DIR, PROMOTIONS_FILE, load_card, read_champion
from prism.router import Router

MODES = {
    "Shadow — the LLM answers, the model is recorded (trial)": "shadow",
    "Gated — the model when sure, the LLM otherwise": "gated",
    "Model only — no LLM (testing)": "model_only",
}

EXAMPLES = [
    "",
    "is the login bug fixed yet?",
    "what did we decide in monday's meeting about the launch?",
    "not the email, what did people say in slack about the outage?",
    "does the ticket match the spec doc for the export feature?",
    "how do I reset my VPN password?",
    "what is a KV cache?",
    "thanks, that was helpful!",
    "look into that thing from earlier",
]

# One colour per source for the chips (styling only)
SOURCE_COLOURS = {
    "jira": ("#E8F0FE", "#1A56DB"),
    "confluence": ("#E6F6FA", "#0E7490"),
    "slack": ("#F3E8FF", "#7E22CE"),
    "notion": ("#F1F2F4", "#374151"),
    "gmail": ("#FDECEC", "#C2410C"),
    "kb": ("#E9F7EF", "#15803D"),
    "web": ("#FFF7E0", "#A16207"),
}
SOURCE_NAMES = {
    "jira": "Jira",
    "confluence": "Confluence",
    "slack": "Slack",
    "notion": "Notion",
    "gmail": "Gmail",
    "kb": "Knowledge Base",
    "web": "Web",
}

# Plain names for the model recipes (the folder codes mean nothing to colleagues)
RECIPE_PLAIN_NAMES = {
    "tfidf_logreg": "Words only",
    "bge_logreg": "Meaning model only",
    "tfidf_bge_logreg": "Words + meaning model",
    "ft_bge_logreg": "Fine-tuned meaning model only",
    "tfidf_ft_bge_logreg": "Words + fine-tuned meaning model",
    "tfidf_ftv2_bge_logreg": "Words + fine-tuned meaning model (with negation examples)",
    "tfidf_ftsplit_bge_logreg": "Words + fine-tuned meaning model (trained on half the data)",
}

# Fixed results of finished comparisons (not re-run): results/laya_comparison.md and
# results/laya_full_system.md. Our own model's numbers come from models/champion.json.
LLM_ONLY_REAL_WORLD = 0.746
LAYA_WITH_LLM_REAL_WORLD = 0.822

REPORTS = {
    "All small models compared": "results/model_comparison.md",
    "Cost vs accuracy (choosing the gate)": "results/cost_accuracy_curve.md",
    "Why not Laya (a bigger ready-made model)": "results/laya_comparison.md",
    "Laya vs our model, each with the LLM": "results/laya_full_system.md",
}

STYLE = """
<style>
.brand {letter-spacing: 0.32em; font-weight: 700; font-size: 1.05rem; margin-bottom: 0;}
.tagline {color: #5B6474; margin-top: 0.1rem;}
.section {letter-spacing: 0.18em; font-size: 0.78rem; color: #5B6474; font-weight: 600;
          margin: 1.2rem 0 0.4rem 0;}
.chip {display: inline-block; padding: 0.35rem 0.8rem; margin: 0 0.4rem 0.4rem 0; border-radius: 999px;
       font-weight: 600; font-size: 0.95rem;}
.chip-none {background: #F4F5F8; color: #5B6474; font-weight: 500;}
.badge {display: inline-block; padding: 0.15rem 0.6rem; border-radius: 6px; font-size: 0.8rem;
        font-weight: 600; margin-left: 0.4rem;}
.badge-model {background: #E9F7EF; color: #15803D;}
.badge-llm {background: #FFF4E5; color: #B45309;}
.panel {background: #F8F9FB; border: 1px solid #ECEEF2; border-radius: 12px; padding: 1rem 1.1rem;}
.muted {color: #5B6474; font-size: 0.88rem;}
</style>
"""


def plain_model_name(card: dict) -> str:
    """'tfidf_ftv2_bge_logreg' → 'Words + fine-tuned meaning model (...)' (+ the negation rule)."""
    name = RECIPE_PLAIN_NAMES.get(card["recipe"], card["recipe"])
    if card["settings"].get("negation_to_llm"):
        name = name + " + negation rule"
    return name


def model_rows(live_version: str) -> list:
    """One row per saved model, read from its card (models/<version>/card.json)."""
    rows = []
    for folder in sorted(MODELS_DIR.iterdir()):
        if not (folder / "card.json").exists():
            continue
        card = load_card(folder.name)
        scores = card["eval"]
        if "coverage" not in scores:  # early versions saved before the gate existed
            continue
        live = ""
        if folder.name == live_version:
            live = "⭐ live"
        rows.append(
            {
                "": live,
                "Model": plain_model_name(card),
                "Right on real-world questions": scores["real_world_exact"] * 100,
                "Right on all questions": scores["all_exact"] * 100,
                "Answers without the LLM": scores["coverage"] * 100,
                "Right when it answers": scores["kept_accuracy"] * 100,
                "Code": folder.name,
            }
        )
    return rows


def promotion_history() -> list:
    """Every change of live model, newest first (models/promotions.jsonl)."""
    rows = []
    history_file = MODELS_DIR / PROMOTIONS_FILE
    if not history_file.exists():
        return rows
    for line in history_file.read_text().splitlines():
        if not line.strip():
            continue
        change = json.loads(line)
        rows.append(
            {
                "When": change["when"],
                "By": change["by"],
                "From": change["from"] or "—",
                "To": change["to"],
                "Why": change["why"],
            }
        )
    rows.reverse()
    return rows


@st.cache_resource  # made ONCE when the app starts, then shared by every user (~36 s to load)
def get_router() -> Router:
    return Router(log=DecisionLog())


def chips(sources) -> str:
    """Coloured chips for a set of sources (HTML). Formatting only."""
    if sources is None:
        return '<span class="chip chip-none">— not asked</span>'
    if not sources:
        return '<span class="chip chip-none">No search needed</span>'
    parts = []
    for source in sorted(sources):
        background, colour = SOURCE_COLOURS.get(source, ("#F4F5F8", "#1F2430"))
        name = html.escape(SOURCE_NAMES.get(source, source))
        parts.append(
            f'<span class="chip" style="background:{background};color:{colour}">{name}</span>'
        )
    return "".join(parts)


def section(title: str) -> None:
    st.markdown(f'<div class="section">{title}</div>', unsafe_allow_html=True)


# ---------- page setup ----------
st.set_page_config(page_title="Aetherion Prism", page_icon="🔷", layout="wide")
st.markdown(STYLE, unsafe_allow_html=True)
router = get_router()
champion = read_champion()

# Remember this tab's numbers between clicks (Streamlit re-runs the script on every click)
if "history" not in st.session_state:
    st.session_state["history"] = []  # this tab's records, newest last

st.markdown('<p class="brand">AETHERION PRISM</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Where should I search? A small model answers when it is sure; '
    "an LLM answers the rest.</p>",
    unsafe_allow_html=True,
)

ask_tab, models_tab, session_tab, how_tab = st.tabs(
    ["Ask", "Models", "This session", "How it works"]
)

# ---------- Ask ----------
with ask_tab:
    left, right = st.columns([3, 2])
    with left:
        example = st.selectbox("Try an example (optional)", EXAMPLES, index=0)
        question = st.text_area(
            "Your question", value=example, placeholder="e.g. is the login bug fixed?", height=90
        )
    with right:
        mode_label = st.radio("Mode", list(MODES.keys()), index=0)  # Shadow is the default
        st.caption("Shadow is the trial mode: please keep it unless you're testing.")

    clicked = st.button("Route question", type="primary", width="stretch")

    if clicked and question.strip():
        record = router.route(question.strip(), mode=MODES[mode_label])
        st.session_state["history"].append(record)

    if st.session_state["history"]:
        record = st.session_state["history"][-1]

        section("SEARCH IN")
        if record.decided_by == "model":
            badge = '<span class="badge badge-model">decided by the model</span>'
        else:
            badge = '<span class="badge badge-llm">decided by the LLM</span>'
        st.markdown(
            f"{chips(record.answer)}{badge}"
            f'<div class="muted">reason: {html.escape(record.reason)} · mode: {record.mode}</div>',
            unsafe_allow_html=True,
        )

        col_answers, col_numbers = st.columns([3, 2])
        with col_answers:
            section("MODEL VS LLM")
            agree = record.model_and_llm_agree()
            if agree is None:
                agree_text = "The LLM wasn't asked."
            elif agree:
                agree_text = "✅ They agree."
            else:
                agree_text = "⚠️ They disagree."
            if record.would_escalate:
                gated_would = "ask the LLM"
            else:
                gated_would = "let the model answer"
            model_title = "Model"
            if record.decided_by != "model":
                model_title = "Model (recorded only)"
            st.markdown(
                f'<div class="panel"><div class="muted">{model_title} · confidence '
                f"{record.model_confidence:.2f} (gate {record.gate})</div>{chips(record.model_answer)}"
                f'<div class="muted" style="margin-top:0.6rem">LLM</div>{chips(record.llm_answer)}'
                f'<div class="muted" style="margin-top:0.6rem">{agree_text} '
                f"Gated mode would {gated_would}."
                "</div></div>",
                unsafe_allow_html=True,
            )
        with col_numbers:
            section("THIS QUESTION")
            m1, m2 = st.columns(2)
            m1.metric("Time", f"{record.latency_ms:,.0f} ms")
            m2.metric("Cost", f"${record.cost_spent:.5f}")
            m1.metric("Saved vs LLM", f"${record.money_saved():.5f}")
            m2.metric("Confidence", f"{record.model_confidence:.2f}")
            if record.llm_error:
                st.warning(f"The LLM failed ({record.llm_error}); the backup answer was used.")

# ---------- Models ----------
with models_tab:
    evidence = champion["evidence_on_eval"]
    live_card = load_card(champion["version"])

    section("WHY THIS MODEL")
    st.markdown(f"**{plain_model_name(live_card)}** · `{champion['version']}`")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("Right on real-world questions", f"{evidence['real_world_exact']:.0%}")
    w2.metric("Answered without the LLM", f"{evidence['model_answers_alone']:.0%}")
    w3.metric("Cheaper than LLM-only", f"{evidence['cheaper_than_llm_only']:.0%}")
    w4.metric("Model + LLM, real-world", f"{evidence['full_system_real_world_exact']:.1%}")
    st.info(champion["why"])
    st.caption(
        f"What it is: {champion['what_it_is']}. Measured on 1,200 test questions the model never "
        f"trained on. Live since {champion['since']}, chosen by {champion['promoted_by']}."
    )

    section("ALL MODELS WE TRIED")
    table = pd.DataFrame(model_rows(champion["version"]))
    # best on real-world first; a tie → the one that is more often right when it answers
    table = table.sort_values(
        ["Right on real-world questions", "Right when it answers"], ascending=False
    )
    percent = st.column_config.NumberColumn(format="%.1f%%")
    st.dataframe(
        table,
        hide_index=True,
        width="stretch",
        column_config={
            "Right on real-world questions": percent,
            "Right on all questions": percent,
            "Answers without the LLM": percent,
            "Right when it answers": percent,
        },
    )
    st.caption(
        "Right on all questions includes the hard cases (47% of the test on purpose). "
        "Answers without the LLM = how often the model is sure enough to answer by itself."
    )
    st.bar_chart(
        table,
        x="Model",
        y="Right on real-world questions",
        horizontal=True,
        sort="-Right on real-world questions",
        color="#E5574B",
    )

    section("MODEL + LLM TOGETHER (REAL-WORLD QUESTIONS)")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("LLM alone", f"{LLM_ONLY_REAL_WORLD:.1%}")
    t2.metric("Our model alone", f"{evidence['real_world_exact']:.1%}")
    t3.metric("Our model + LLM", f"{evidence['full_system_real_world_exact']:.1%}")
    t4.metric("Laya + LLM", f"{LAYA_WITH_LLM_REAL_WORLD:.1%}")
    st.markdown(
        "**Why not Laya** (a bigger ready-made model, fine-tuned on our data)? With the LLM it "
        "was only a tie overall, slightly worse on real-world and company questions, "
        "**8× slower** (~54 ms vs ~7 ms) and **340× bigger** (1.6 GB). It was better only on "
        "the hard cases, and those already go to the LLM."
    )

    section("HISTORY OF THE LIVE MODEL")
    st.dataframe(promotion_history(), hide_index=True, width="stretch")

    section("FULL REPORTS")
    for title, report_file in REPORTS.items():
        with st.expander(title):
            st.markdown(Path(report_file).read_text())

# ---------- This session ----------
with session_tab:
    history = st.session_state["history"]
    asked = len(history)
    by_model = 0
    llm_calls = 0
    cost = 0.0
    saved = 0.0
    for record in history:  # adding up the records' own numbers (no routing logic)
        if record.decided_by == "model":
            by_model += 1
        if record.llm_answer is not None or record.llm_error is not None:
            llm_calls += 1
        cost += record.cost_spent
        saved += record.money_saved()
    s1, s2, s3, s4 = st.columns(4)
    s1.metric("Questions", asked)
    s2.metric("Answered by the model", by_model)
    s3.metric("Cost", f"${cost:.4f}")
    s4.metric("Saved vs LLM every time", f"${saved:.4f}")
    st.caption(f"LLM calls: {llm_calls} · your browser tab only")
    if history:
        rows = []
        for record in reversed(history):
            rows.append(
                {
                    "question": record.question,
                    "search in": ", ".join(sorted(record.answer)) or "no search",
                    "decided by": record.decided_by,
                    "reason": record.reason,
                    "time (ms)": round(record.latency_ms),
                    "cost ($)": round(record.cost_spent, 5),
                }
            )
        st.dataframe(rows, width="stretch", hide_index=True)

# ---------- How it works ----------
with how_tab:
    st.markdown(
        """
**The idea:** asking an LLM where to search, for every question, is slow (~1–2 s) and costs money.
Most questions are easy, so a small model answers those in milliseconds, for free, and only
the uncertain ones go to the LLM.

**For every question:**
1. The small model scores 7 sources (Jira, Confluence, Slack, Notion, Gmail, Knowledge Base, Web)
   and says how sure it is (its *confidence*).
2. Questions with "not", "except" or "skip" always go to the LLM (the model is weak at negation).
3. If the confidence is above the *gate*, the model answers; otherwise the LLM answers.
4. Every decision is written to the decision log.

**The three modes:**
- **Shadow** (the trial): the LLM always answers you; the model's answer is only recorded.
  After enough real questions, a person checks the disagreements, and the model may go live
  only if it's right on at least 85% of the questions it's sure about.
- **Gated** (after go-live): the model answers when it's sure, the LLM otherwise.
- **Model only** (testing): no LLM at all.

**Results on 1,200 test questions:** real-world accuracy 84% (better than the LLM alone, 75%),
76% cheaper than asking the LLM every time.
"""
    )

# ---------- Sidebar ----------
with st.sidebar:
    section("LIVE MODEL")
    st.markdown(f"`{champion['version']}`")
    st.caption(f"since {champion['since']} · {champion['what_it_is']}")
    section("SETUP")
    st.markdown(
        f"Default mode: **{router.mode}**  \nLLM: `{router.settings.llm_model}` · prompt "
        f"{router.settings.llm_prompt_version}  \nGate: {router.gate}"
    )
    section("SHADOW TRIAL")
    goal = router.settings.go_live_min_questions  # 500, from the YAML
    done = count_shadow_questions()
    st.markdown(f"**{done} / {goal}** real questions")
    st.progress(min(done / goal, 1.0))
    st.caption("Needed before the go-live check.")
