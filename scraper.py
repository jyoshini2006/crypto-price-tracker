"""Stage 2 + 3 - resilient Selenium scraping of CoinMarketCap.

The fragile part of the project, handled defensively:

* driver creation is wrapped in a context manager so Chrome always quits,
* the table is waited for with ``WebDriverWait`` (never ``time.sleep``),
* the page is scrolled once to trigger lazily rendered rows,
* ``__NEXT_DATA__`` (the JSON blob Next.js embeds in the page) is preferred over
  CSS selectors because it survives markup reshuffles,
* every attempt is retried with exponential backoff and the failure is raised as
  :class:`ScrapeError` so the caller can fall back to the API.
"""

from __future__ import annotations

import json
import logging
import random
import re
import time
from contextlib import contextmanager
from typing import Any, Iterable, Iterator, Mapping

from cleaner import parse_market_cap, parse_name, parse_number, parse_percent, parse_price, parse_symbol

LOGGER = logging.getLogger(__name__)

_SYMBOL_RE = re.compile(r"^[A-Z0-9]{2,10}$")
_PERCENT_RE = re.compile(r"^[+\-\u2212]?\d[\d,]*\.?\d*\s*%$")
_MONEY_RE = re.compile(r"^[$\u20ac\u00a3\u00a5]\s?[\d,]+\.?\d*\s*[KMBT]?$", re.IGNORECASE)
_NAME_WITH_SYMBOL_RE = re.compile(r"^(?P<name>.+?)\s*\((?P<symbol>[A-Za-z0-9]{2,10})\)\s*$")

#: Key names that different CoinMarketCap / CoinGecko payloads use for the same field.
_PRICE_KEYS = ("price", "current_price", "priceUsd", "price_usd", "last_price")
_CHANGE_KEYS = (
    "percent_change_24h",
    "percentChange24h",
    "price_change_percentage_24h",
    "priceChangePercentage24h",
    "change24h",
    "change_24h",
)
_MARKET_CAP_KEYS = ("market_cap", "marketCap", "market_cap_usd", "marketCapUsd")
_NAME_KEYS = ("name", "coinName", "title")
_SYMBOL_KEYS = ("symbol", "ticker", "assetSymbol")
_QUOTE_KEYS = ("quote", "quotes", "USD", "usd")


class ScrapeError(RuntimeError):
    """Raised when Selenium could not produce a usable batch."""


# ---------------------------------------------------------------------------
# payload parsing (shared with the CoinMarketCap API fallback)
# ---------------------------------------------------------------------------
def _lookup(mapping: Mapping[str, Any], keys: Iterable[str], max_depth: int = 3) -> Any:
    """Find the first matching key, descending into nested dicts a few levels."""
    frontier: list[tuple[Mapping[str, Any], int]] = [(mapping, 0)]
    seen: set[int] = set()
    while frontier:
        node, depth = frontier.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        for key in keys:
            if key in node and not isinstance(node[key], (dict, list)):
                return node[key]
        if depth < max_depth:
            for value in node.values():
                if isinstance(value, dict):
                    frontier.append((value, depth + 1))
    return None


def _looks_like_listing(node: Mapping[str, Any]) -> bool:
    symbol = parse_symbol(_lookup(node, _SYMBOL_KEYS, max_depth=1))
    if not symbol or not _SYMBOL_RE.match(symbol):
        return False
    price = parse_price(_lookup(node, _PRICE_KEYS, max_depth=2))
    return price is not None and price > 0


def extract_listings_from_payload(payload: Any, top_n: int | None = None) -> list[dict[str, Any]]:
    """Recursively harvest coin records from an arbitrary nested JSON payload.

    Works for the ``__NEXT_DATA__`` blob embedded in the CoinMarketCap page and
    for the CoinMarketCap Pro API response, without depending on either layout.
    """
    found: list[dict[str, Any]] = []
    seen_symbols: set[str] = set()

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            if _looks_like_listing(node):
                symbol = parse_symbol(_lookup(node, _SYMBOL_KEYS, max_depth=1))
                if symbol and symbol not in seen_symbols:
                    seen_symbols.add(symbol)
                    found.append(
                        {
                            "name": parse_name(_lookup(node, _NAME_KEYS, max_depth=2)) or symbol,
                            "symbol": symbol,
                            "price": parse_price(_lookup(node, _PRICE_KEYS, max_depth=2)),
                            "change_24h": parse_percent(_lookup(node, _CHANGE_KEYS, max_depth=3)),
                            "market_cap": parse_market_cap(_lookup(node, _MARKET_CAP_KEYS, max_depth=3)),
                            "source": "selenium",
                        }
                    )
                return
            for value in node.values():
                visit(value)
        elif isinstance(node, list):
            for item in node:
                visit(item)

    visit(payload)
    return found[:top_n] if top_n else found


def _text_of(element: Any) -> str:
    try:
        return (element.text or "").strip()
    except Exception:  # pragma: no cover - stale elements are retried upstream
        return ""


def _cell_texts(row: Any) -> list[str]:
    try:
        cells = row.find_elements("css selector", "td, [role='cell']")
    except Exception:  # pragma: no cover
        return []
    return [text for text in (_text_of(cell) for cell in cells) if text]


# ---------------------------------------------------------------------------
# scraper
# ---------------------------------------------------------------------------
class CoinMarketCapScraper:
    """Selenium client for the CoinMarketCap top-``N`` table."""

    def __init__(self, cfg: Any, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or LOGGER
        self.url: str = cfg.get("scraper.url", "https://coinmarketcap.com/")
        self.top_n: int = int(cfg.get("scraper.top_n", 10))
        self.headless: bool = bool(cfg.get("scraper.headless", True))
        self.window_size: str = cfg.get("scraper.window_size", "1920,1080")
        self.user_agent: str = cfg.get("scraper.user_agent", "")
        self.block_images: bool = bool(cfg.get("scraper.block_images", True))
        self.page_load_timeout: int = int(cfg.get("scraper.page_load_timeout", 45))
        self.element_timeout: int = int(cfg.get("scraper.element_timeout", 25))
        self.scroll_pause: float = float(cfg.get("scraper.scroll_pause", 1.5))
        self.retries: int = max(1, int(cfg.get("scraper.retries", 3)))
        self.backoff_base: float = float(cfg.get("scraper.backoff_base", 2.0))
        self.prefer_embedded_json: bool = bool(cfg.get("scraper.prefer_embedded_json", True))
        self.selectors: dict[str, Any] = cfg.section("scraper").get("selectors", {}) or {}

    # -- driver lifecycle -------------------------------------------------
    def _build_options(self) -> Any:
        from selenium.webdriver.chrome.options import Options

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={self.window_size}")
        options.add_argument("--lang=en-US")
        options.add_argument("--disable-blink-features=AutomationControlled")
        if self.user_agent:
            options.add_argument(f"--user-agent={self.user_agent}")
        if self.block_images:
            # Two independent switches: prefs for content settings, blink for layout.
            options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
            options.add_argument("--blink-settings=imagesEnabled=false")
        options.add_experimental_option("excludeSwitches", ["enable-automation", "enable-logging"])
        return options

    def _create_driver(self) -> Any:
        """Create a Chrome driver, preferring webdriver_manager for the binary."""
        from selenium import webdriver

        options = self._build_options()
        service = None
        try:
            from webdriver_manager.chrome import ChromeDriverManager
            from selenium.webdriver.chrome.service import Service

            service = Service(ChromeDriverManager().install())
        except Exception as exc:  # pragma: no cover - offline / no manager available
            # Selenium 4.6+ can resolve the driver itself (Selenium Manager).
            self.log.warning("webdriver_manager unavailable (%s); using Selenium Manager", exc)

        if service is not None:
            return webdriver.Chrome(service=service, options=options)
        return webdriver.Chrome(options=options)

    @contextmanager
    def driver(self) -> Iterator[Any]:
        """Yield a Chrome driver and always quit it, even on a crash."""
        driver = self._create_driver()
        try:
            yield driver
        finally:
            try:
                driver.quit()
            except Exception:  # pragma: no cover - already dead
                self.log.debug("driver.quit() failed", exc_info=True)

    # -- page interaction -------------------------------------------------
    def _wait_for_rows(self, driver: Any) -> list[Any]:
        from selenium.webdriver.support.ui import WebDriverWait

        selectors = self.selectors.get("rows") or ["table tbody tr"]

        def rows_present(current: Any) -> list[Any] | bool:
            for selector in selectors:
                try:
                    rows = current.find_elements("css selector", selector)
                except Exception:  # pragma: no cover - mid-navigation
                    continue
                if rows:
                    return rows
            return False

        return WebDriverWait(driver, self.element_timeout).until(rows_present)

    def _scroll_once(self, driver: Any) -> None:
        try:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight / 3);")
            time.sleep(self.scroll_pause)  # let lazy rows render after the scroll
            driver.execute_script("window.scrollTo(0, 0);")
        except Exception:  # pragma: no cover
            self.log.debug("scroll failed", exc_info=True)

    def _load(self, driver: Any) -> None:
        driver.set_page_load_timeout(self.page_load_timeout)
        driver.get(self.url)
        self._wait_for_rows(driver)
        self._scroll_once(driver)

    # -- extraction -------------------------------------------------------
    def _rows_from_embedded_json(self, driver: Any) -> list[dict[str, Any]]:
        from selenium.common.exceptions import NoSuchElementException

        try:
            script = driver.find_element("css selector", "script#__NEXT_DATA__")
        except NoSuchElementException:
            return []
        payload = script.get_attribute("innerHTML") or ""
        if not payload.strip():
            return []
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            self.log.warning("__NEXT_DATA__ present but not valid JSON")
            return []
        return extract_listings_from_payload(parsed, top_n=self.top_n)

    def _find_text(self, row: Any, key: str) -> str:
        selector = self.selectors.get(key)
        if not selector:
            return ""
        for candidate in str(selector).split(","):
            candidate = candidate.strip()
            if not candidate:
                continue
            try:
                elements = row.find_elements("css selector", candidate)
            except Exception:  # pragma: no cover
                continue
            for element in elements:
                text = _text_of(element)
                if text:
                    return text
        return ""

    def _row_from_dom(self, row: Any) -> dict[str, Any] | None:
        """Best-effort DOM row parse, driven by config selectors plus heuristics."""
        name_text = self._find_text(row, "name")
        symbol_text = self._find_text(row, "symbol") or ""
        price_text = self._find_text(row, "price")
        change_text = self._find_text(row, "change_24h")
        cap_text = self._find_text(row, "market_cap")

        cells = _cell_texts(row)
        for text in cells:
            if not symbol_text and _SYMBOL_RE.match(text):
                symbol_text = text
            elif not price_text and _MONEY_RE.match(text):
                price_text = text
            elif not change_text and _PERCENT_RE.match(text):
                change_text = text
            elif not cap_text and _MONEY_RE.match(text) and text != price_text:
                cap_text = text

        name, symbol = name_text, parse_symbol(symbol_text)
        match = _NAME_WITH_SYMBOL_RE.match(name_text)
        if match:
            name = match.group("name").strip()
            symbol = symbol or parse_symbol(match.group("symbol"))
        elif not symbol:
            # CoinMarketCap renders the name and the ticker as one cell
            # ("Bitcoin BTC") when only the container is matched.
            tokens = name_text.split()
            if len(tokens) >= 2 and _SYMBOL_RE.match(tokens[-1].upper()):
                symbol = parse_symbol(tokens[-1])
                name = " ".join(tokens[:-1])

        price = parse_price(price_text)
        if price is None or price <= 0 or not symbol:
            return None
        return {
            "name": parse_name(name) or symbol,
            "symbol": symbol,
            "price": price,
            "change_24h": parse_percent(change_text),
            "market_cap": parse_market_cap(cap_text),
            "source": "selenium",
        }

    def _rows_from_dom(self, driver: Any) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row in self._wait_for_rows(driver):
            record = self._row_from_dom(row)
            if not record or record["symbol"] in seen:
                continue
            seen.add(record["symbol"])
            records.append(record)
            if len(records) >= self.top_n:
                break
        return records

    def _extract(self, driver: Any) -> list[dict[str, Any]]:
        if self.prefer_embedded_json:
            records = self._rows_from_embedded_json(driver)
            if len(records) >= min(self.top_n, 5):
                self.log.info("parsed %d rows from __NEXT_DATA__", len(records))
                return records
            self.log.info("embedded JSON gave %d rows; falling back to DOM", len(records))

        records = self._rows_from_dom(driver)
        self.log.info("parsed %d rows from the DOM", len(records))
        if not records:
            raise ScrapeError("no rows could be parsed from the CoinMarketCap table")
        return records

    # -- public API -------------------------------------------------------
    def fetch_top(self) -> list[dict[str, Any]]:
        """Scrape the top coins, retrying with exponential backoff."""
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                with self.driver() as driver:
                    self._load(driver)
                    records = self._extract(driver)
                if records:
                    return records
                last_error = ScrapeError("scraper returned zero rows")
            except Exception as exc:  # selenium + parsing + driver errors
                last_error = exc
                self.log.warning(
                    "scrape attempt %d/%d failed: %s: %s",
                    attempt,
                    self.retries,
                    type(exc).__name__,
                    exc,
                )
            if attempt < self.retries:
                delay = self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.75)
                self.log.info("retrying scrape in %.1fs", delay)
                time.sleep(delay)
        raise ScrapeError(f"scraping failed after {self.retries} attempts: {last_error}") from last_error
