# syntax=docker/dockerfile:1.7

FROM ghcr.io/astral-sh/uv:0.11.14 AS uv

FROM python:3.14-slim-bookworm AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

COPY --from=uv /uv /uvx /bin/
COPY pyproject.toml uv.lock ./

RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project


FROM python:3.14-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FRAMESEG_HOST=0.0.0.0 \
    FRAMESEG_PORT=8000 \
    FRAMESEG_DATA_ROOT=/data \
    HF_HOME=/cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1

RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 frameseg \
    && useradd --system --uid 10001 --gid frameseg --home-dir /app frameseg \
    && mkdir -p /data /cache/huggingface \
    && chown -R frameseg:frameseg /data /cache/huggingface

WORKDIR /app

COPY --from=builder --chown=frameseg:frameseg /app/.venv /app/.venv
COPY --chown=frameseg:frameseg *.py ./

USER frameseg

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=5)"]

CMD ["python", "main.py"]
