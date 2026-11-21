# Backend API image. Also contains the coordinator (simulation and distributed
# runners, which the launchers spawn as separate processes) and the site agent
# used by the distributed demo.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first (cached layer), pinned by uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra backend --extra science --extra site --no-install-project

COPY shared ./shared
COPY client ./client
COPY coordinator ./coordinator
COPY backend ./backend
COPY evaluation ./evaluation
COPY harmonization ./harmonization
COPY experiments ./experiments

RUN useradd --create-home app && mkdir -p /app/var && chown -R app /app/var
USER app
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app_factory", "--factory", "--host", "0.0.0.0", "--port", "8000"]
