"""The Aetherion Prism page (Streamlit): a thin WINDOW over router.route().

It decides nothing: it sends the question to the router and shows the DecisionRecord
(and the record's own helpers: money_saved, model_and_llm_agree). All the logic lives in
router.py / decisionRecord.py, so if something looks wrong, debug those, not the page.

Run:  uv run streamlit run src/prism/app.py          (the look: .streamlit/config.toml)
"""

import html
import json
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st

from prism.batch import (
    FILE_TYPES,
    read_questions_file,
    results_rows,
    route_many,
    split_questions,
)
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

# Plain words for the evidence numbers saved in champion.json / promotions.jsonl / model cards
EVIDENCE_LABELS = {
    "real_world_exact": "right on real-world questions",
    "all_exact": "right on all questions",
    "all_needed": "gets every needed source",
    "coverage": "answers without the LLM",
    "model_answers_alone": "answers without the LLM",
    "kept_accuracy": "right when it answers",
    "full_system_all_exact": "model + LLM: right on all questions",
    "full_system_real_world_exact": "model + LLM: right on real-world questions",
    "cheaper_than_llm_only": "cheaper than LLM-only",
}

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
    """Every change of live model (promotions and rollbacks), newest first (promotions.jsonl)."""
    changes = []
    history_file = MODELS_DIR / PROMOTIONS_FILE
    if not history_file.exists():
        return changes
    for line in history_file.read_text().splitlines():
        if line.strip():
            changes.append(json.loads(line))
    changes.reverse()
    return changes


def evidence_text(evidence: dict) -> str:
    """{"real_world_exact": 0.84, ...} → "right on real-world questions 84.0% · ..." """
    parts = []
    for key, value in evidence.items():
        label = EVIDENCE_LABELS.get(key, key)
        if isinstance(value, (int, float)):
            parts.append(f"{label} {value:.1%}")
        else:
            parts.append(f"{label} {value}")
    return " · ".join(parts)


def percent_or_dash(numbers: dict, key: str) -> str:
    """A score as "84.2%", or "—" if this model's evidence doesn't have it (e.g. after a rollback)."""
    if key not in numbers:
        return "—"
    return f"{numbers[key]:.1%}"


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

ask_tab, batch_tab, live_tab, models_tab, session_tab, how_tab = st.tabs(
    ["Ask", "Batch", "Live model", "Models", "This session", "How it works"]
)
live_card = load_card(champion["version"])  # the live model's card: scores, gate, thresholds
evidence = champion["evidence_on_eval"]

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

# ---------- Batch ----------
with batch_tab:
    st.markdown(
        "Route many questions at once. Every question is logged like a single one. "
        "**For training, use Shadow mode:** retraining needs the LLM's answer next to the model's."
    )
    pasted = st.text_area(
        "Paste questions: one per line, or whole paragraphs",
        height=140,
        placeholder="is the login bug fixed?\nwhat did slack say about the outage? check the spec too.",
    )
    uploaded = st.file_uploader(
        "…or upload a file (.txt: one per line · .csv / .xlsx: a 'question' column, or the first column)",
        type=FILE_TYPES,
    )

    batch_questions = []
    try:
        if uploaded is not None:  # a file wins over the pasted text
            batch_questions = read_questions_file(uploaded.name, uploaded.getvalue())
        else:
            batch_questions = split_questions(pasted)
    except (ValueError, BadZipFile) as error:  # an unreadable file: say why, route nothing
        st.error(f"Couldn't read the file: {error}")

    limit = router.settings.batch_max_questions
    if batch_questions:
        st.markdown(f"Found **{len(batch_questions)}** questions. Check them before routing:")
        preview = []
        for number in range(len(batch_questions)):
            preview.append({"#": number + 1, "question": batch_questions[number]})
        st.dataframe(preview, hide_index=True, width="stretch", height=180)
        if len(batch_questions) > limit:
            st.error(
                f"That's more than the limit of {limit} (batch_max_questions). Split the file."
            )

    batch_mode_label = st.radio("Mode", list(MODES.keys()), index=0, key="batch_mode")
    if MODES[batch_mode_label] != "model_only":
        st.caption("Each question that goes to the LLM takes ~1.5 s and costs ~$0.002.")
    ready = len(batch_questions) > 0 and len(batch_questions) <= limit
    if st.button("Route all", type="primary", width="stretch", disabled=not ready):
        progress_bar = st.progress(0.0, text="Starting…")

        def show_progress(done, total):
            progress_bar.progress(done / total, text=f"Routed {done} / {total}")

        st.session_state["batch_records"] = route_many(
            router,
            batch_questions,
            limit,
            mode=MODES[batch_mode_label],
            on_progress=show_progress,
        )

    if st.session_state.get("batch_records"):
        batch_records = st.session_state["batch_records"]
        batch_by_model = 0
        batch_cost = 0.0
        batch_saved = 0.0
        for record in batch_records:  # adding up the records' own numbers
            if record.decided_by == "model":
                batch_by_model += 1
            batch_cost += record.cost_spent
            batch_saved += record.money_saved()
        section("RESULTS")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Questions", len(batch_records))
        b2.metric("Answered by the model", batch_by_model)
        b3.metric("Cost", f"${batch_cost:.4f}")
        b4.metric("Saved vs LLM every time", f"${batch_saved:.4f}")
        rows = results_rows(batch_records)
        st.dataframe(rows, hide_index=True, width="stretch")
        st.download_button(
            "Download results (CSV)",
            data=pd.DataFrame(rows).to_csv(index=False),
            file_name="routing_results.csv",
            mime="text/csv",
        )

# ---------- Live model ----------
with live_tab:
    if router.model_version != champion["version"]:  # promoted after the page started
        st.warning(
            f"A new model was promoted ({champion['version']}), but this page still uses "
            f"{router.model_version}. Restart the page to use the new one."
        )

    section("ANSWERING QUESTIONS NOW")
    st.markdown(f"#### {plain_model_name(live_card)}")
    st.caption(
        f"`{champion['version']}` · live since {champion['since']} · "
        f"chosen by {champion['promoted_by']}"
    )
    st.info(f"**Why:** {champion['why']}")
    st.caption(f"What it is: {champion['what_it_is']}.")

    scores = live_card["eval"]
    section("ON ITS OWN (1,200 TEST QUESTIONS IT NEVER TRAINED ON)")
    o1, o2, o3 = st.columns(3)
    o1.metric("Right on real-world questions", percent_or_dash(scores, "real_world_exact"))
    o2.metric("Right on all questions", percent_or_dash(scores, "all_exact"))
    o3.metric("Gets every needed source", percent_or_dash(scores, "all_needed"))
    st.caption("All questions include the hard cases (47% of the test on purpose).")

    section("THE GATE (WHEN IT ANSWERS BY ITSELF)")
    gate_slice = live_card["gate_slice"]
    g1, g2, g3 = st.columns(3)
    g1.metric("Gate", live_card["gate"])
    g2.metric("Answers without the LLM", percent_or_dash(scores, "coverage"))
    g3.metric("Right when it answers", percent_or_dash(scores, "kept_accuracy"))
    st.caption(
        f"The gate was chosen on separate practice questions (the gate slice) so the model is right "
        f"on at least {live_card['settings']['accuracy_floor']:.0%} of the questions it keeps: "
        f"there it kept {gate_slice['coverage']:.1%} and was right on {gate_slice['accuracy']:.1%}. "
        "Below the gate (or a question with not / except / skip) → the LLM answers."
    )

    section("WITH THE LLM (THE FULL SYSTEM)")
    f1, f2, f3 = st.columns(3)
    f1.metric(
        "Right on real-world questions", percent_or_dash(evidence, "full_system_real_world_exact")
    )
    f2.metric("Right on all questions", percent_or_dash(evidence, "full_system_all_exact"))
    f3.metric("Cheaper than LLM-only", percent_or_dash(evidence, "cheaper_than_llm_only"))

    section("PER SOURCE (WHEN EACH YES/NO JUDGE SAYS YES)")
    source_rows = []
    for source, threshold in live_card["thresholds"].items():
        source_rows.append(
            {"Source": SOURCE_NAMES.get(source, source), "Says yes above": threshold}
        )
    st.dataframe(source_rows, hide_index=True, width="stretch")
    st.caption(
        "Each source has its own judge giving a score from 0 to 1; the bar was chosen per source "
        "on practice questions (the threshold slice), not on the test."
    )

    section("HISTORY OF CHANGES (NEWEST FIRST)")
    for change in promotion_history():
        if change["from"] is None:
            title = f"**{change['when']}** · first live model: **{change['to']}**"
        else:
            title = f"**{change['when']}** · {change['from']} → **{change['to']}**"
        with st.container(border=True):
            st.markdown(f"{title} · by {change['by']}")
            st.markdown(f"**Why:** {change['why']}")
            st.caption(evidence_text(change["evidence_on_eval"]))

    section("HOW TO CHANGE IT")
    st.markdown(
        """
1. **Collect** reviewed real questions from a shadow run:
   `uv run python -m prism.retrain collect logs/<run folder>`
2. **Train and compare** a new version with the live one on the same test:
   `uv run python -m prism.retrain train` → it can be promoted only if it is not worse
   (overall, real-world, and confident mistakes on hard questions) and all tests pass.
3. **Promote** it (a person decides, and must write why):
   `uv run python -m prism.retrain promote <version> "<what it is>" "<why>"` → restart the page.

Changed your mind? `uv run python -m prism.retrain rollback "<why>"` goes back to the previous model
(it shows up in the history above too). Nothing is ever promoted automatically.
"""
    )

# ---------- Models ----------
with models_tab:
    st.caption("Every model we trained, compared on the same 1,200 test questions.")
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
    t2.metric("Our model alone", percent_or_dash(live_card["eval"], "real_world_exact"))
    t3.metric("Our model + LLM", percent_or_dash(evidence, "full_system_real_world_exact"))
    t4.metric("Laya + LLM", f"{LAYA_WITH_LLM_REAL_WORLD:.1%}")
    st.markdown(
        "**Why not Laya** (a bigger ready-made model, fine-tuned on our data)? With the LLM it "
        "was only a tie overall, slightly worse on real-world and company questions, "
        "**8× slower** (~54 ms vs ~7 ms) and **340× bigger** (1.6 GB). It was better only on "
        "the hard cases, and those already go to the LLM."
    )

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
