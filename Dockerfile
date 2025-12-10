FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd -m appuser
WORKDIR /app

# uv pour les dépendances
RUN pip install --no-cache-dir uv

# Dépendances (cache de build) : pyproject + uv.lock d'abord
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

ENV PATH="/app/.venv/bin:${PATH}"

# Code
COPY . .

# Variables pour Spaces
ENV PORT=7860
EXPOSE 7860

USER appuser
CMD ["sh","-lc","uvicorn src.api.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
