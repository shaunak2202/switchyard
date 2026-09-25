# One image for both the gateway and the mock provider; compose picks the command.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies in their own layer so source edits don't reinstall them.
COPY pyproject.toml README.md ./
RUN mkdir -p switchyard mock_provider \
 && touch switchyard/__init__.py mock_provider/__init__.py \
 && pip install . \
 && pip uninstall -y switchyard

COPY switchyard ./switchyard
COPY mock_provider ./mock_provider
RUN pip install --no-deps .
COPY config ./config

RUN useradd --uid 10001 --no-create-home app
USER app

EXPOSE 8000
CMD ["uvicorn", "switchyard.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--loop", "uvloop", "--http", "httptools"]
