# University Info Chatbot

A **RAG chatbot** that scrapes university websites, indexes them, and answers questions based on actual information. Scrape interval is configurable through a web interface.

## Architecture

```
scraper/   — Crawls university sites, extracts clean text (polite scraping)
indexer/   — Chunking, local embeddings, vector DB (ChromaDB), deduplication
api/       — FastAPI: /ask (RAG), /universities, /settings, /scrape, /health
ui/        — Web interface: chat + scrape interval settings via UI
config/    — universities.yaml (universities, intervals, model parameters)
data/      — Vector database and page hashes (auto-generated)
```

## Quick Start (local)

```bash
cd university_chatbot
pip install -r requirements.txt

# Ollama must be running with a generation model:
ollama pull qwen2.5-coder:7b

# Run
python -m api.main
```

Open `http://localhost:8000/ui` — chat and settings.

## Docker

```bash
docker compose up -d --build
```

## API

| Method | Path | What it does |
|---|---|---|
| GET | `/health` | liveness check |
| GET | `/universities` | list of universities + intervals |
| POST | `/settings` | change scrape interval |
| POST | `/scrape` | manual scrape trigger |
| POST | `/ask` | question → answer + sources |

## How to Add a University

1. Open `config/universities.yaml`
2. Add a block (copy existing one)
3. Restart the service — university appears in UI.

## Features

- Local embeddings (sentence-transformers) — no external API costs
- Local generation (Ollama) — data stays on your machine
- Deduplication by hash — repeated scrapes do not duplicate pages
- Answers strictly from context; if no data is found, the bot honestly says "I don't know"
