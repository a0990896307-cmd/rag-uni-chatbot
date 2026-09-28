FROM python:3.11-slim

WORKDIR /app

# системные зависимости для sentence-transformers / chroma
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scraper/ ./scraper/
COPY indexer/ ./indexer/
COPY api/ ./api/
COPY ui/ ./ui/
COPY config/ ./config/

# healthcheck: сервис жив
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s \
  CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]