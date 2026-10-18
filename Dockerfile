# syntax=docker/dockerfile:1
FROM python:3.12.12-slim-bookworm AS build
COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /build
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY scripts/download_model.py ./scripts/
RUN --mount=type=cache,target=/root/.cache/huggingface /opt/venv/bin/python scripts/download_model.py
COPY app/__init__.py app/model.py ./app/
# Loading offline during the build catches missing model/tokenizer files.
RUN HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 /opt/venv/bin/python -c \
    "from app.model import load_model; load_model('/opt/models/gliner2.5-decide', 'cpu', 'fp32')"

FROM python:3.12.12-slim-bookworm AS runtime

ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MODEL_PATH=/opt/models/gliner2.5-decide \
    DEVICE=cuda \
    PRECISION=fp16 \
    IDLE_TIMEOUT_SECONDS=180 \
    OMP_NUM_THREADS=2

RUN useradd --create-home --uid 10001 choice
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/models /opt/models
WORKDIR /app
COPY app/ ./app/
ENV FORWARDED_ALLOW_IPS=*
USER choice
EXPOSE 8000
CMD ["python", "-m", "app"]
