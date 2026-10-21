FROM python:3.10-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
WORKDIR /app

# CPU build of the torch version in poetry.lock; PyPI would serve the CUDA one.
RUN PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu pip install torch==2.14.1+cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install .

COPY artifacts ./artifacts
RUN useradd -m app
USER app

EXPOSE 8000
CMD ["python", "-m", "ship_routing.service"]
