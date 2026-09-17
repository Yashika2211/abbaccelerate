FROM python:3.11-slim

# libgomp1: OpenMP runtime required by LightGBM and XGBoost.
# curl: container healthcheck.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Dependency layer first so source edits do not re-resolve the world.
COPY pyproject.toml ./
RUN uv pip install --system -r pyproject.toml --extra dev

COPY . .

EXPOSE 8000
CMD ["uvicorn", "kairos.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
