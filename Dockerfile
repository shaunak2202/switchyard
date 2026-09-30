# Two targets from one file:
#   mock    - the mock provider (small: FastAPI only)
#   gateway - the gateway. With --build-arg SEMANTIC=true it also bakes in CPU-only torch and
#             the semantic-cache models (the tier is off by default, ADR-021), so it starts
#             without network access and without a first-request download.
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

RUN useradd --uid 10001 --create-home app

# ---------------------------------------------------------------------------------------------
FROM base AS mock
COPY switchyard ./switchyard
COPY mock_provider ./mock_provider
RUN pip install --no-deps .
USER app
EXPOSE 9000
CMD ["uvicorn", "mock_provider.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "9000", "--no-access-log", "--log-level", "warning", "--timeout-keep-alive", "75", "--loop", "uvloop", "--http", "httptools"]

# ---------------------------------------------------------------------------------------------
FROM base AS gateway
ARG SEMANTIC=false
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ARG VERIFIER_MODEL=cross-encoder/quora-distilroberta-base
ENV HF_HOME=/opt/hf
# CPU wheels only: the default PyPI torch for x86_64 bundles CUDA (~2 GB) we would never use.
RUN if [ "$SEMANTIC" = "true" ]; then \
      pip install torch --index-url https://download.pytorch.org/whl/cpu \
      && pip install "sentence-transformers>=3.3" \
      && python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('${EMBEDDING_MODEL}', device='cpu'); CrossEncoder('${VERIFIER_MODEL}', device='cpu')" \
      && chmod -R a+rX /opt/hf; \
    fi
# Never reach out to the Hub at runtime.
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

COPY switchyard ./switchyard
COPY mock_provider ./mock_provider
RUN pip install --no-deps .
COPY config ./config

USER app
EXPOSE 8000
CMD ["uvicorn", "switchyard.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--no-access-log", "--timeout-keep-alive", "75", "--loop", "uvloop", "--http", "httptools"]
