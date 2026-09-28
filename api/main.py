"""FastAPI backend: /ask (RAG), /universities, /settings + background re-scraper."""
import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from scraper.scraper import UniScraper
from indexer.indexer import Indexer

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")


def _clean_think(text: str) -> str:
    """Удаляет блок </think>... — внутренние размышления reasoning-моделей."""
    import re
    # Разные форматы:  response,  thought,  response
    text = re.sub(r"<\|?(begin|start)_?(of_)?(thinking|thought)\|?>.*?<\|?(end|stop)_?(of_)?(thinking|thought)\|?>",
                  "", text, flags=re.DOTALL | re.IGNORECASE)
    # Если остался обычный "thinking" в начале
    return text.strip()

BASE_DIR = Path(__file__).resolve().parent.parent

# ── Зависимости инициализируются лениво (эмбеддинги грузятся тяжело) ──
embedder = None
llm_client = None
llm_model = "deepseek-r1:8b"
indexers = {}
universities = {}
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 3


def load_universities():
    global universities, llm_model
    import yaml
    with open(BASE_DIR / "config" / "universities.yaml") as f:
        cfg = yaml.safe_load(f)
    universities = {u["id"]: u for u in cfg["universities"]}
    llm_model = cfg.get("llm", {}).get("model", llm_model)


def get_embedder():
    """Локальные эмбеддинги: sentence-transformers (бесплатно, оффлайн)."""
    global embedder
    if embedder is None:
        from sentence_transformers import SentenceTransformer
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        log.info("Загрузка эмбеддера %s ...", model_name)
        embedder = SentenceTransformer(model_name)
    return embedder


def get_llm():
    """Генерация ответов через локальную Ollama (приватно, бесплатно)."""
    global llm_client
    if llm_client is None:
        import requests
        # проверим доступность
        requests.get("http://localhost:11434/api/tags", timeout=3)
        llm_client = "ollama"
    return llm_client


def get_indexer(uni_id):
    if uni_id in indexers:
        return indexers[uni_id]
    idx = Indexer(uni_id)
    indexers[uni_id] = idx
    return idx


def run_scrape(uni_id: str):
    uni = universities[uni_id]
    scraper = UniScraper(
        base_url=uni["base_url"],
        max_pages=uni.get("max_pages", 200),
        allowed_paths=uni.get("allowed_paths"),
    )
    pages = scraper.scrape()
    embedder = get_embedder()
    idx = get_indexer(uni_id)
    stats = idx.update(pages, embedder.encode, CHUNK_SIZE, CHUNK_OVERLAP)
    log.info("Индексация %s: %s", uni_id, stats)
    return stats


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Загружаем конфиг; НЕ запускаем фоновый скрейп автоматически.
    # Скрейп выполняется по запросу через /scrape или вручную.
    load_universities()
    yield


app = FastAPI(title="University Info Chatbot", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# статика (фронтенд)
app.mount("/ui", StaticFiles(directory=str(BASE_DIR / "ui"), html=True), name="ui")


class AskRequest(BaseModel):
    question: str
    university: str = None


@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@app.get("/universities")
def list_universities():
    return [{"id": u["id"], "name": u["name"], "base_url": u["base_url"],
             "scrape_interval_days": u["scrape_interval_days"]} for u in universities.values()]


class SettingsUpdate(BaseModel):
    university: str
    scrape_interval_days: int


@app.post("/settings")
def update_settings(body: SettingsUpdate):
    """Меняем интервал скрейпинга для конкретного университета (через UI)."""
    if body.university not in universities:
        raise HTTPException(404, "университет не найден")
    if not (1 <= body.scrape_interval_days <= 60):
        raise HTTPException(400, "интервал 1–60 дней")
    universities[body.university]["scrape_interval_days"] = body.scrape_interval_days
    return {"ok": True, "university": body.university,
            "scrape_interval_days": body.scrape_interval_days}


@app.post("/scrape")
def scrape_now(body: dict):
    """Ручной запуск скрейпинга для университета."""
    uni_id = body.get("university")
    if uni_id not in universities:
        raise HTTPException(404, "университет не найден")
    stats = run_scrape(uni_id)
    return {"ok": True, "stats": stats}


@app.post("/ask")
def ask(body: AskRequest):
    """RAG-ответ: эмбеддинг вопроса → поиск чанков → генерация через Ollama."""
    embed_fn = get_embedder()
    idx = get_indexer(body.university) if body.university else None

    # фильтр по университету на уровне извлечения
    uni_filter = body.university if body.university and body.university in universities else None

    q_emb = embed_fn.encode([body.question]).tolist()[0]
    results = []
    if idx is not None:
        results = idx.query(q_emb, TOP_K, uni_filter)
    elif universities:
        # ищем по всем индексам
        for uid in universities:
            r = indexers[uid].query(q_emb, TOP_K, uid) if uid in indexers else []
            results.extend(r)

    if not results:
        return {"answer": "Sorry, I couldn't find information on this question. "
                          "Please try rephrasing or specify the university.",
                 "sources": []}
    results.sort(key=lambda x: x["distance"])

    context = "\n\n".join(f"[{r['metadata'].get('title','')}]({r['metadata'].get('url','')}):\n{r['text']}"
                          for r in results[:TOP_K])

    model = llm_model  # из конфига universities.yaml
    prompt = f"""You are a university information assistant. Answer ONLY based on the context below.
If the answer is not in the context — honestly say you don't know.
Answer concisely and to the point.

CONTEXT:
{context}

QUESTION: {body.question}

ANSWER:"""

    import requests, json as _json
    try:
        resp = requests.post("http://localhost:11434/api/generate", json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,          # deepseek-r1: выключаем длинные размышления
            "options": {"num_predict": 300},  # ограничиваем длину ответа
        }, timeout=300)
        resp.raise_for_status()
        answer = resp.json()["response"].strip()
        # Убираем блок "thinking" у reasoning-моделей (deepseek-r1 и т.п.)
        answer = _clean_think(answer)
    except Exception as e:
        log.error("Ollama error: %s", e)
        answer = "Не удалось сгенерировать ответ (проблема с локальной моделью)."

    sources = [{"url": r["metadata"].get("url"), "title": r["metadata"].get("title")}
               for r in results[:TOP_K]]
    return {"answer": answer, "sources": sources}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)