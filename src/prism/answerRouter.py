"""Router 2: the answer door. Prism's prediction for one question the platform sent to /intake.

GET /prediction/{question_id}
    → {"question_id": "108", "status": "predicted", "prism_answer": ["jira"], "model_sure": true,
       "confidence": 0.91, "model_version": "...", "predicted_at": "..."}
    status "waiting" = saved, but the model hasn't answered yet (it's retried on the next send)
    404 = no question with that id was ever sent

Only Prism's prediction goes out: the platform already has the question and its intent answer.
"""

from fastapi import APIRouter, HTTPException, Request

answer_router = APIRouter()


@answer_router.get("/prediction/{question_id}")
def prediction(question_id: str, request: Request):
    found = request.app.state.store.get_prediction(question_id)
    if found is None:
        raise HTTPException(status_code=404, detail=f"No question with id {question_id!r}")

    if found["prism_answer"] is None:
        status = "waiting"
    else:
        status = "predicted"

    return {
        "question_id": question_id,
        "status": status,
        "prism_answer": found["prism_answer"],
        "model_sure": found["model_sure"],
        "confidence": found["confidence"],
        "model_version": found["model_version"],
        "predicted_at": found["predicted_at"],
    }
