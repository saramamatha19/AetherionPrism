"""The API: a second door into the SAME router, for PROGRAMS (the Streamlit page is for people).

POST /route   {"question": "...", "mode": "shadow" | "gated" | "model_only" (optional)}
              → the DecisionRecord as JSON (and the decision is logged, as always)
GET  /health  → is it running, which model is live, the default mode

Run:  uv run uvicorn prism.api:app --port 8000
Try:  http://localhost:8000/docs   (FastAPI makes a test page automatically)
"""

from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from prism.decisionLog import DecisionLog
from prism.router import Router


class RouteRequest(BaseModel):
    """What a program sends. A wrong mode word → FastAPI refuses it with a clear error."""

    question: str
    mode: Literal["shadow", "gated", "model_only"] | None = None  # None = the default from the YAML


def create_app(make_router=None) -> FastAPI:
    """Build the API. Tests pass their own router (fake LLM, no real log); normally the real one."""
    if make_router is None:

        def make_router():
            return Router(log=DecisionLog())

    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state["router"] = make_router()  # load ONCE when the API starts (~36 s)
        yield

    app = FastAPI(title="Aetherion Prism", lifespan=lifespan)

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
