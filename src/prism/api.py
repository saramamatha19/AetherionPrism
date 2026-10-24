"""The API: a second door into the SAME router, for PROGRAMS (the Streamlit page is for people).

POST /route   {"question": "...", "mode": "shadow" | "gated" | "model_only" (optional)}
              → the DecisionRecord as JSON (and the decision is logged, as always)
GET  /health  → is it running, which model is live, the default mode
POST /intake  → Router 1: real questions from the Aetherion platform's cron (see intakeRouter.py)
GET  /prediction/{question_id} → Router 2: Prism's prediction for one of them (see answerRouter.py)

Run:  uv run uvicorn prism.api:app --port 8000
Try:  http://localhost:8000/docs   (FastAPI makes a test page automatically)
PRISM_MODE=model_only (an environment variable) replaces the YAML mode: Docker uses it, so the
container needs no API key.
"""

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prism.answerRouter import answer_router
from prism.config import load_settings
from prism.decisionLog import DecisionLog
from prism.intakeRouter import intake_router
from prism.intakeStore import IntakeStore
from prism.router import Router


class RouteRequest(BaseModel):
    """What a program sends. A wrong mode word → FastAPI refuses it with a clear error."""

    question: str
    mode: Literal["shadow", "gated", "model_only"] | None = None  # None = the default from the YAML


def load_api_settings():
    """The YAML settings, but PRISM_MODE (an environment variable) can replace the mode.
    Docker sets PRISM_MODE=model_only: the model only, never the LLM, so no API key is needed."""
    settings = load_settings()
    mode = os.environ.get("PRISM_MODE")
    if mode is None:
        return settings
    if mode not in ("model_only", "gated", "shadow"):
        raise ValueError(f"PRISM_MODE={mode!r}: use model_only, gated or shadow")
    return settings.model_copy(update={"mode": mode})


def create_app(make_router=None, make_store=None) -> FastAPI:
    """Build the API. Tests pass their own router (fake LLM, no real log) and store (a temporary
    file); normally the real ones."""
    if make_router is None:

        def make_router():
            return Router(settings=load_api_settings(), log=DecisionLog())

    if make_store is None:
        make_store = IntakeStore  # intake/prism.db

    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["router"] = make_router()  # load ONCE when the API starts (~36 s)
        app.state.router = state["router"]  # the routers in other files find them here
        app.state.store = make_store()
        yield

    app = FastAPI(title="Aetherion Prism", lifespan=lifespan)
    app.include_router(intake_router)  # Router 1: POST /intake
    app.include_router(answer_router)  # Router 2: GET /prediction/{question_id}

    @app.get("/health")
    def health():
        router = state["router"]
        return {"status": "ok", "model": router.model_version, "default_mode": router.mode}

    @app.post("/route")
    def route(request: RouteRequest):
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="The question is empty")
        record = state["router"].route(question, mode=request.mode)
        return record.to_dict()

    return app


app = create_app()  # what uvicorn runs
