"""Scraping layer: crawl university sites, extract clean text.
Safe defaults: rate limiting, timeouts, max pages, same-domain only.
"""
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

log = logging.getLogger("scraper")

# Блокируемые расширения (не-HTML файлы)
SKIP_EXTS = (
    ".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg", ".zip", ".rar",
    ".tar", ".gz", ".mp4", ".mp3", ".doc", ".docx", ".xls", ".xlsx",
    ".ppt", ".pptx", ".css", ".js", ".woff", ".woff2", ".ico", ".xml",
)

# Элементы, которые выкидываем из текста (навигация, футеры, скрипты)
REMOVE_TAGS = {"script", "style", "noscript", "header", "footer", "nav", "aside"}


@dataclass
class PageResult:
    url: str
    title: str
    text: str
    fetched_at: str
    content_hash: str
    links: list = field(default_factory=list)


class UniScraper:
    def __init__(self, base_url: str, max_pages: int = 200,
                 allowed_paths=None, delay: float = 0.7, timeout: int = 20):
        self.base_url = base_url.rstrip("/")
        self.base_host = urlparse(self.base_url).netloc
        self.max_pages = max_pages
        self.allowed_paths = allowed_paths
        self.delay = delay          # пауза между запросами (вежливый скрейпинг)
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "UniversityInfoBot/1.0 (research; contact: admin@example.com)",
        })

    def _same_host(self, url: str) -> bool:
        try:
            return urlparse(url).netloc == self.base_host
        except Exception:
            return False

    def _in_allowed_path(self, url: str) -> bool:
        if not self.allowed_paths:
            return True
        path = urlparse(url).path
        return any(path.startswith(p) for p in self.allowed_paths)

    def _is_skippable(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return path.endswith(SKIP_EXTS) or "#" in url

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, timeout=self.timeout, allow_redirects=True)
            resp.raise_for_status()
        except Exception as e:
            log.warning("Ошибка %s: %s", url, e)
            return None
        # Только HTML-ответы
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype:
            return None
        return BeautifulSoup(resp.text, "html.parser")

    @staticmethod
    def _clean_text(soup: BeautifulSoup) -> str:
        for tag in REMOVE_TAGS:
            for el in soup.find_all(tag):
                el.decompose()
        # Заголовки страницы
        title = ""
        if soup.title:
            title = soup.title.get_text(strip=True)
        text = soup.get_text(separator="\n", strip=True)
        # Схлопываем пустые строки
        text = re.sub(r"\n{2,}", "\n", text)
        return title, text

    def scrape(self) -> list[PageResult]:
        """Обход сайта BFS/DFS с ограничением числа страниц."""
        results: dict[str, PageResult] = {}
        seen: set[str] = set()
        queue = [self.base_url]

        while queue and len(results) < self.max_pages:
            url = queue.pop(0)
            if url in seen or not self._same_host(url) or not self._in_allowed_path(url):
                continue
            if self._is_skippable(url):
                continue
            seen.add(url)

            soup = self._fetch(url)
            if soup is None:
                continue

            title, text = self._clean_text(soup)
            if not text.strip():
                continue

            content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

            # Собираем внутренние ссылки
            links = []
            for a in soup.find_all("a", href=True):
                href = a["href"].strip()
                full = urljoin(url, href)
                if self._same_host(full) and self._in_allowed_path(full) and not self._is_skippable(full):
                    links.append(full)
            # Дедупликация и обрезка
            links = list(dict.fromkeys(links))[:200]

            results[url] = PageResult(
                url=url,
                title=title,
                text=text,
                fetched_at=datetime.utcnow().isoformat(),
                content_hash=content_hash,
                links=links,
            )
            log.info("[%d/%d] %s", len(results), self.max_pages, url)

            # добавляем новые ссылки в очередь
            for l in links:
                if l not in seen:
                    queue.append(l)

            time.sleep(self.delay)  # вежливость к серверу

        return list(results.values())


def split_chunks(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Делит текст на перекрывающиеся чанки по словам (для эмбеддингов)."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks