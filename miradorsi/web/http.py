"""Client HTTP de Mirador SI : cache disque, politesse, robots.txt, limites de taille.

Uniquement la bibliothèque standard (urllib). Les variables HTTPS_PROXY /
HTTP_PROXY et SSL_CERT_FILE de l'environnement sont respectées.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import ssl
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib import robotparser

from ..config import WebConfig


class WebError(Exception):
    """Erreur réseau ou HTTP (non fatale pour une analyse)."""


class HttpStatusError(WebError):
    """Réponse HTTP en erreur ; `retry_after` vient de l'en-tête Retry-After (secondes)."""

    def __init__(self, status: int, url: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status} sur {url}")
        self.status = status
        self.retry_after = retry_after

    @property
    def retryable(self) -> bool:
        return self.status == 429 or self.status >= 500


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None  # format date HTTP : on se rabat sur le délai exponentiel


@dataclass
class HttpResponse:
    url: str
    status: int
    headers: dict[str, str] = field(default_factory=dict)
    body: bytes = b""
    from_cache: bool = False

    @property
    def content_type(self) -> str:
        return self.headers.get("content-type", "").split(";")[0].strip().lower()

    @property
    def text(self) -> str:
        return self.body.decode(self._charset(), errors="replace")

    def json(self):
        return json.loads(self.text)

    def _charset(self) -> str:
        match = re.search(r"charset=([\w-]+)", self.headers.get("content-type", ""), re.I)
        if not match:
            match = re.search(rb"<meta[^>]+charset=[\"']?([\w-]+)", self.body[:4096], re.I)
            name = match.group(1).decode("ascii", "ignore") if match else "utf-8"
        else:
            name = match.group(1)
        try:
            "".encode(name)
            return name
        except LookupError:
            return "utf-8"


class WebClient:
    def __init__(self, cfg: WebConfig | None = None):
        self.cfg = cfg or WebConfig()
        self.cache_dir = Path(self.cfg.cache_dir) if self.cfg.cache_dir else None
        self._last_hit: dict[str, float] = {}
        self._host_delay: dict[str, float] = {}  # délais allongés après un « trop de requêtes »
        self._robots: dict[str, robotparser.RobotFileParser | None] = {}
        self._lock = threading.Lock()
        self._ssl = ssl.create_default_context()
        self.opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=self._ssl))

    # ------------------------------------------------------------------ API
    def get(self, url: str, params: dict | None = None, headers: dict | None = None,
            use_cache: bool = True, check_robots: bool = True) -> HttpResponse:
        if params:
            url = f"{url}{'&' if '?' in url else '?'}{urllib.parse.urlencode(params)}"
        scheme = urllib.parse.urlsplit(url).scheme
        if scheme not in ("http", "https"):
            raise WebError(f"Schéma non supporté : {url}")

        if use_cache:
            cached = self._cache_read(url)
            if cached is not None:
                return cached
        if check_robots and self.cfg.respect_robots_txt and not self.allowed_by_robots(url):
            raise WebError(f"Accès refusé par robots.txt : {url}")

        last_error: Exception | None = None
        host = urllib.parse.urlsplit(url).netloc
        for attempt in range(self.cfg.retries + 1):
            try:
                self._wait_politely(url)
                response = self._request(url, headers or {})
                if use_cache and response.status == 200:
                    self._cache_write(url, response)
                return response
            except HttpStatusError as exc:
                last_error = exc
                if not exc.retryable or attempt == self.cfg.retries:
                    break
                wait = exc.retry_after if exc.retry_after is not None else 2.0 * 2 ** attempt
                if wait > self.cfg.max_retry_wait_seconds:
                    break  # le serveur demande d'attendre trop longtemps : on abandonne poliment
                if exc.status == 429:  # trop de requêtes : on ralentit durablement pour ce site
                    self._host_delay[host] = min(30.0, max(self._delay_for(host) * 2, 2.0))
                time.sleep(wait)
            except (WebError, urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                last_error = exc
                if attempt < self.cfg.retries:
                    time.sleep(min(2 ** attempt, 8) * 0.5)
        raise WebError(f"Échec de {url} : {last_error}")

    # ------------------------------------------------------------ interne
    def _request(self, url: str, headers: dict) -> HttpResponse:
        all_headers = {
            "User-Agent": self.cfg.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/json;q=0.9,text/plain;q=0.8,*/*;q=0.5",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
        }
        all_headers.update(headers)
        request = urllib.request.Request(url, headers=all_headers)
        try:
            with self.opener.open(request, timeout=self.cfg.timeout_seconds) as resp:
                body = resp.read(self.cfg.max_bytes + 1)
                status = resp.status
                final_url = resp.geturl()
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
        except urllib.error.HTTPError as err:
            retry_after = _parse_retry_after(err.headers.get("Retry-After") if err.headers else None)
            raise HttpStatusError(err.code, url, retry_after) from err
        if len(body) > self.cfg.max_bytes:
            body = body[: self.cfg.max_bytes]
        encoding = resp_headers.get("content-encoding", "").lower()
        try:
            if encoding == "gzip":
                body = gzip.decompress(body)
            elif encoding == "deflate":
                body = zlib.decompress(body)
        except (OSError, zlib.error, EOFError):
            pass  # contenu tronqué : on garde les octets bruts
        return HttpResponse(final_url, status, resp_headers, body)

    def _delay_for(self, host: str) -> float:
        return self._host_delay.get(host, self.cfg.min_delay_seconds)

    def _wait_politely(self, url: str) -> None:
        host = urllib.parse.urlsplit(url).netloc
        with self._lock:
            now = time.monotonic()
            wait = self._last_hit.get(host, 0.0) + self._delay_for(host) - now
            self._last_hit[host] = max(now, now + wait)
        if wait > 0:
            time.sleep(wait)

    def allowed_by_robots(self, url: str) -> bool:
        parts = urllib.parse.urlsplit(url)
        base = f"{parts.scheme}://{parts.netloc}"
        if base not in self._robots:
            parser: robotparser.RobotFileParser | None = robotparser.RobotFileParser()
            try:
                resp = self.get(base + "/robots.txt", check_robots=False)
                if resp.status == 200:
                    parser.parse(resp.text.splitlines())
                else:
                    parser = None
            except WebError:
                parser = None  # robots.txt absent ou injoignable : autorisé
            self._robots[base] = parser
        parser = self._robots[base]
        return True if parser is None else parser.can_fetch(self.cfg.user_agent, url)

    # ------------------------------------------------------------ cache
    def _cache_path(self, url: str) -> Path | None:
        if not self.cache_dir:
            return None
        return self.cache_dir / (hashlib.sha256(url.encode("utf-8")).hexdigest()[:32] + ".json")

    def _cache_read(self, url: str) -> HttpResponse | None:
        path = self._cache_path(url)
        if path is None or not path.exists():
            return None
        if time.time() - path.stat().st_mtime > self.cfg.cache_ttl_hours * 3600:
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return HttpResponse(data["url"], data["status"], data["headers"],
                                data["body"].encode("latin-1"), from_cache=True)
        except (OSError, ValueError, KeyError):
            return None

    def _cache_write(self, url: str, response: HttpResponse) -> None:
        path = self._cache_path(url)
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {"url": response.url, "status": response.status, "headers": response.headers,
                    "body": response.body.decode("latin-1")}
            path.write_text(json.dumps(data), encoding="utf-8")
        except OSError:
            pass
