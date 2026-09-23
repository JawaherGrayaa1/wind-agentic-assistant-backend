from __future__ import annotations

import html
import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx


class _PageParser(HTMLParser):
    """Small dependency-free parser for readable page text and HTML tables."""

    _ignored_tags = {"script", "style", "noscript", "svg", "template"}
    _block_tags = {
        "address", "article", "aside", "blockquote", "br", "dd", "div",
        "dl", "dt", "footer", "h1", "h2", "h3", "h4", "h5", "h6",
        "header", "li", "main", "nav", "ol", "p", "pre", "section",
        "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.text_parts: list[str] = []
        self.title_parts: list[str] = []
        self._title_depth = 0
        self._ignored_depth = 0
        self.tables: list[list[list[str]]] = []
        self._table: list[list[str]] | None = None
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    @staticmethod
    def _clean(value: str) -> str:
        return re.sub(r"\s+", " ", html.unescape(value)).strip()

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        if tag == "title":
            self._title_depth += 1
        if self._ignored_depth == 0 and tag in self._block_tags:
            self.text_parts.append("\n")
        if self._ignored_depth == 0 and tag == "table" and self._table is None:
            self._table = []
        if self._ignored_depth == 0 and tag == "tr" and self._table is not None:
            self._row = []
        if self._ignored_depth == 0 and tag in {"td", "th"} and self._row is not None:
            self._cell = []

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._ignored_depth == 0 and tag in {"td", "th"} and self._row is not None and self._cell is not None:
            self._row.append(self._clean(" ".join(self._cell)))
            self._cell = None
        if self._ignored_depth == 0 and tag == "tr" and self._table is not None and self._row is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        if self._ignored_depth == 0 and tag == "table" and self._table is not None:
            if self._table:
                self.tables.append(self._table)
            self._table = None
        if tag == "title" and self._title_depth:
            self._title_depth -= 1
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        value = self._clean(data)
        if not value:
            return
        self.text_parts.append(value)
        if self._title_depth:
            self.title_parts.append(value)
        if self._cell is not None:
            self._cell.append(value)


class WebScraper:
    """Fetch public HTML pages and normalize their useful text/table content."""

    def __init__(self, timeout: float = 30.0, max_response_bytes: int = 4_000_000):
        self.timeout = max(5.0, min(float(timeout), 60.0))
        self.max_response_bytes = max(100_000, int(max_response_bytes))

    @staticmethod
    def _validate_url(url: str, allow_private_network: bool) -> str:
        parsed = urlparse(url.strip())
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("Only absolute HTTP(S) URLs are supported.")
        if parsed.username or parsed.password:
            raise ValueError("URLs containing embedded credentials are not supported.")
        if allow_private_network:
            return url.strip()
        hostname = parsed.hostname
        try:
            addresses = {item[4][0] for item in socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)}
        except socket.gaierror as exc:
            raise ValueError(f"Could not resolve hostname '{hostname}'.") from exc
        for address in addresses:
            ip = ipaddress.ip_address(address)
            if any((ip.is_private, ip.is_loopback, ip.is_link_local, ip.is_reserved, ip.is_multicast, ip.is_unspecified)):
                raise ValueError("Private or local network targets are blocked by default.")
        return url.strip()

    def scrape(
        self,
        url: str,
        report_request: str = "",
        max_chars: int = 12_000,
        max_table_rows: int = 100,
        allow_private_network: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        try:
            current_url = self._validate_url(url, allow_private_network)
            max_chars = max(1_000, min(int(max_chars), 30_000))
            max_table_rows = max(1, min(int(max_table_rows), 250))
            headers = {
                "Accept": "text/html,application/xhtml+xml,text/plain;q=0.8,*/*;q=0.5",
                "User-Agent": "Wind-Agentic-WebScraper/1.0",
            }
            with httpx.Client(timeout=self.timeout, headers=headers) as client:
                response: httpx.Response | None = None
                for _redirect in range(5):
                    current_url = self._validate_url(current_url, allow_private_network)
                    response = client.get(current_url, follow_redirects=False)
                    if response.status_code not in {301, 302, 303, 307, 308}:
                        break
                    location = response.headers.get("location")
                    if not location:
                        break
                    current_url = urljoin(current_url, location)
                if response is None:
                    raise ValueError("The page did not return a response.")
                response.raise_for_status()
                content_type = response.headers.get("content-type", "").lower()
                if content_type and not any(kind in content_type for kind in ("html", "xhtml", "text/plain")):
                    raise ValueError(f"Unsupported content type: {content_type.split(';', 1)[0]}")
                raw = response.content
                if len(raw) > self.max_response_bytes:
                    raw = raw[: self.max_response_bytes]
                source_text = raw.decode(response.encoding or "utf-8", errors="replace")

            parser = _PageParser()
            parser.feed(source_text)
            parser.close()
            text = re.sub(r"\n{3,}", "\n\n", " ".join(parser.text_parts)).strip()
            text = text[:max_chars]
            tables = [
                [row[:50] for row in table[:max_table_rows]]
                for table in parser.tables[:20]
                if table
            ]
            return {
                "found": True,
                "source": "web-scraper",
                "source_url": current_url,
                "title": " ".join(parser.title_parts).strip() or None,
                "text": text,
                "tables": tables,
                "truncated": len(text) >= max_chars or len(raw) >= self.max_response_bytes,
                "status_code": response.status_code,
                "content_type": content_type or "text/html",
                "report_request": report_request,
            }
        except (httpx.HTTPError, ValueError, UnicodeError, OSError) as exc:
            return {
                "found": False,
                "source": "web-scraper",
                "source_url": url,
                "error": str(exc),
                "message": "The web page could not be scraped.",
            }
