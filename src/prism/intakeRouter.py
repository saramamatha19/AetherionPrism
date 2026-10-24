"""Router 1: the intake door. The Aetherion platform's cron sends real questions here.

POST /intake  {"questions": [{"question_id": "108", "question": "...", "intent_answer": ["jira"]}]}
              → {"saved": 1, "skipped_duplicates": 0, "skipped_empty": 0, "predicted": 1}

1. SAVE every question first (so the data is safe even if the model fails)
2. PREDICT every question the model hasn't answered yet (this send + any that failed before)
The platform's source names are saved EXACTLY as sent; they're translated only when comparing.
Only the model runs here (model_only): the platform's intent prompt is already the LLM side.
"""

import logging

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

intake_router = APIRouter()


class IntakeQuestion(BaseModel):
    """One question from the platform. A missing or empty question_id → refused (a cron bug)."""

    question_id: str = Field(min_length=1)
    question: str  # no length limit: clients ask however they ask
    intent_answer: list[str]  # the platform's sources, its own names; [] = no source


class IntakeRequest(BaseModel):
    """What the cron sends: a list inside {"questions": ...}, so fields can be added later."""

    questions: list[IntakeQuestion]


def predict_unpredicted(store, router) -> int:
    """The model answers every saved question it hasn't answered yet. One failure never stops
    the rest: that question stays unpredicted and is tried again on the next send."""
    predicted = 0
    for item in store.get_unpredicted():
        try:
            record = router.route(item["question"], write_log=False, mode="model_only")
            store.save_prediction(
                item["question_id"],
                sorted(record.model_answer),
                not record.would_escalate,  # sure = gated mode would let the model answer alone
                record.model_confidence,
                record.model_version,
            )
            predicted += 1
        except Exception:
            logger.exception("Prediction failed for question_id %s", item["question_id"])
    return predicted


@intake_router.post("/intake")
def intake(body: IntakeRequest, request: Request):
    store = request.app.state.store
    router = request.app.state.router

    limit = router.settings.intake_max_questions
    if len(body.questions) > limit:
        raise HTTPException(
            status_code=400,
            detail=f"Too many questions in one send: {len(body.questions)} (max {limit})",
        )

    # 1. Save first
    saved = 0
    skipped_duplicates = 0
    skipped_empty = 0
    for item in body.questions:
        question = item.question.strip()
        if not question:
            skipped_empty += 1  # nothing to route; skipped, but counted so it's never silent
        elif store.save_question(item.question_id, question, item.intent_answer):
            saved += 1
        else:
            skipped_duplicates += 1

    # 2. Then predict
    predicted = predict_unpredicted(store, router)

    return {
        "saved": saved,
        "skipped_duplicates": skipped_duplicates,
        "skipped_empty": skipped_empty,
        "predicted": predicted,
    }
