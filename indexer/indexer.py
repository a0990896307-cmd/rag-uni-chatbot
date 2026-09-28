"""Indexing layer: chunk text -> embeddings -> vector store (ChromaDB).
Uses local sentence-transformers for embeddings (no external API costs).
"""
import hashlib
import json
import logging
import os
from pathlib import Path

import chromadb

from scraper.scraper import PageResult, split_chunks

log = logging.getLogger("indexer")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMA_DIR = DATA_DIR / "chroma"
HASH_DB = DATA_DIR / "seen_hashes.json"


class Indexer:
    def __init__(self, uni_id: str, collection_name: str = "uni_pages"):
        self.uni_id = uni_id
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_or_create_collection(
            collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._load_hashes()

    # ── дедупликация: чтобы повторные скрейпы не задублировали страницы ──
    def _load_hashes(self):
        if HASH_DB.exists():
            self.hashes = json.loads(HASH_DB.read_text())
        else:
            self.hashes = {}
        # убираем хэши других университетов
        self.hashes = self.hashes.get(self.uni_id, {})

    def _save_hashes(self):
        all_hashes = {}
        if HASH_DB.exists():
            all_hashes = json.loads(HASH_DB.read_text())
        all_hashes[self.uni_id] = self.hashes
        HASH_DB.write_text(json.dumps(all_hashes, ensure_ascii=False, indent=2))

    def update(self, pages: list[PageResult], embed_fn, chunk_size: int = 800,
               chunk_overlap: int = 100) -> dict:
        """Обновляет индекс: новые/изменённые страницы добавляются, старые не трогаем."""
        added = updated = skipped = 0
        batch_ids, batch_docs, batch_metas = [], [], []

        for page in pages:
            if page.content_hash in self.hashes:
                skipped += 1
                continue
            chunks = split_chunks(page.text, chunk_size, chunk_overlap)
            for ci, chunk in enumerate(chunks):
                cid = hashlib.sha256(f"{page.url}::{ci}".encode()).hexdigest()[:24]
                batch_ids.append(cid)
                batch_docs.append(chunk)
                batch_metas.append({
                    "url": page.url,
                    "title": page.title[:200],
                    "uni_id": self.uni_id,
                    "chunk_idx": ci,
                    "fetched_at": page.fetched_at,
                })
            # помечаем хэш (если страница изменилась при следующем скрейпе — пере-индекс)
            self.hashes[page.content_hash] = page.url
            added += 1

        if batch_ids:
            embeddings = embed_fn(batch_docs)
            self.collection.add(ids=batch_ids, documents=batch_docs,
                                embeddings=embeddings, metadatas=batch_metas)
            self._save_hashes()

        return {"added": added, "skipped": skipped, "updated": updated}

    def query(self, embedding, top_k: int = 5, uni_filter: str = None):
        """Поиск ближайших чанков по вектору."""
        where = {"uni_id": uni_filter} if uni_filter else None
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where,
        )
        out = []
        for i in range(len(res["ids"][0])):
            out.append({
                "id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "distance": res["distances"][0][i],
                "metadata": res["metadatas"][0][i],
            })
        return out