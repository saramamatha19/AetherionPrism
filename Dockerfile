# Prism in a box: the API (POST /intake, GET /prediction, /health, /route) with the live model.
# No secrets inside: it runs the model only (PRISM_MODE=model_only), so it needs no API key.
#
# Build (on the laptop, where finetuned/ is):  docker build --platform linux/amd64 -t prism:v1 .
# Run:   docker run -p 8000:8000 -v <server folder>:/app/intake prism:v1
#        the -v folder keeps prism.db (the real questions) safe when the container is replaced

# 1. A small Linux with Python 3.12 (the same version as .python-version)
FROM python:3.12-slim

# 2. uv, the same installer and version as on the laptop
COPY --from=ghcr.io/astral-sh/uv:0.12.2 /uv /usr/local/bin/uv

WORKDIR /app

# 3. The libraries first, exactly as in uv.lock (CPU-only PyTorch on Linux; no test tools).
#    Docker reuses this slow step when only the code changes, so rebuilds are fast.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

# 4. The code, the settings, and everything the live model loads at start-up.
#    finetuned/bge_ft_v2 = the meaning model the live model uses (models/champion.json).
#    If a new champion uses a different finetuned/ folder, change it here AND in .dockerignore.
COPY src/ src/
COPY configs/ configs/
COPY models/ models/
COPY finetuned/bge_ft_v2/ finetuned/bge_ft_v2/
COPY recordings/ recordings/
RUN uv sync --frozen --no-dev

# 5. Settings for running
ENV PATH="/app/.venv/bin:$PATH"
ENV PRISM_MODE=model_only
# Never download models while running: everything the model needs is already inside
ENV HF_HUB_OFFLINE=1

# 6. The folder for prism.db. The volume is connected here when the container runs.
RUN mkdir -p intake

# 7. Start the API. 0.0.0.0 = accept connections from outside the container.
EXPOSE 8000
CMD ["uvicorn", "prism.api:app", "--host", "0.0.0.0", "--port", "8000"]
