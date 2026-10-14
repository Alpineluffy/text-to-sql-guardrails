# One image for every role: database setup, API, UI and evaluation.
#   docker compose up --build             (see docker-compose.yml)
#   docker build --target test .          (runs the test suite inside the image)
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
# The embedding model (PyTorch, ~2 GB) is left out; the schema filter is
# skipped in the container, which costs little with a handful of tables.
# anthropic stays on the 0.x line the code is tested with.
RUN pip install ".[web,postgres]" "anthropic>=0.125,<1"

COPY knowledge ./knowledge
COPY config ./config
COPY ui ./ui
COPY docker/entrypoint.sh /usr/local/bin/entrypoint
RUN chmod +x /usr/local/bin/entrypoint \
 && useradd --create-home app \
 && mkdir -p /app/cache \
 && chown -R app /app/cache /app/knowledge
USER app

ENTRYPOINT ["entrypoint"]
CMD ["api"]


# Build-time test run on the container's Python (3.12).
FROM app AS test
USER root
RUN pip install "pytest>=7"
COPY tests ./tests
RUN chown -R app /app
USER app
RUN python -m pytest -q -p no:cacheprovider
