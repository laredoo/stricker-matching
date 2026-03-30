FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY apps/model/pyproject.toml /app/apps/model/pyproject.toml
COPY apps/model/src /app/apps/model/src

RUN pip install --upgrade pip \
    && pip install /app/apps/model

RUN mkdir -p /app/data/artifacts /app/data/output

EXPOSE 8001

CMD ["python", "-m", "stricker_matching_model.main", "server", "--host", "0.0.0.0", "--port", "8001"]
