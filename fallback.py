"""Stage 4 - fallback data sources.

When Selenium fails after its retries the pipeline must still produce a batch.
Providers are tried in the order configured (``fallback.providers``) and every
record they return is tagged ``source="api"`` so the stored history makes it
obvious where a row came from.

CoinGecko needs no key at all; the CoinMarketCap provider is only used when
``CMC_API_KEY`` is present. ``sample_records`` is a deterministic offline source
used by ``--source sample`` for demos and tests.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Mapping

import requests

LOGGER = logging.getLogger(__name__)


class FallbackError(RuntimeError):
    """Raised when no fallback provider could deliver a batch."""


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number


class _Provider:
    """Common retry/backoff behaviour for HTTP providers."""

    name = "provider"

    def __init__(self, cfg: Any, session: requests.Session | None = None, logger: logging.Logger | None = None) -> None:
        self.cfg = cfg
        self.log = logger or LOGGER
        self.session = session or requests.Session()
        self.timeout: float = float(cfg.get("fallback.timeout", 20))
        self.retries: int = max(1, int(cfg.get("fallback.retries", 2)))
        self.backoff_base: float = float(cfg.get("fallback.backoff_base", 2.0))

    def available(self) -> bool:
        return True

    def fetch_top(self, top_n: int) -> list[dict[str, Any]]:  # pragma: no cover - interface
        raise NotImplementedError

    def _get(self, url: str, **kwargs: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            try:
                response = self.session.get(url, timeout=self.timeout, **kwargs)
                response.raise_for_status()
                return response.json()
            except Exception as exc:
                last_error = exc
                self.log.warning(
                    "%s request %d/%d failed: %s", self.name, attempt, self.retries, exc
                )
                if attempt < self.retries:
                    time.sleep(self.backoff_base * (2 ** (attempt - 1)) + random.uniform(0, 0.5))
        raise FallbackError(f"{self.name} request failed: {last_error}") from last_error


class CoinGeckoProvider(_Provider):
    """Free, key-less market data (the default fallback)."""

    name = "coingecko"

    def fetch_top(self, top_n: int) -> list[dict[str, Any]]:
        base = str(self.cfg.get("fallback.coingecko_base_url", "https://api.coingecko.com/api/v3")).rstrip("/")
        payload = self._get(
            f"{base}/coins/markets",
            params={
                "vs_currency": self.cfg.get("fallback.vs_currency", "usd"),
                "order": "market_cap_desc",
                "per_page": max(1, int(top_n)),
                "page": 1,
                "sparkline": "false",
                "price_change_percentage": "24h",
            },
            headers={"Accept": "application/json"},
        )
        if not isinstance(payload, list):
            raise FallbackError(f"unexpected CoinGecko payload: {type(payload).__name__}")

        records: list[dict[str, Any]] = []
        for entry in payload[:top_n]:
            if not isinstance(entry, Mapping):
                continue
            symbol = str(entry.get("symbol") or "").upper()
            price = _to_float(entry.get("current_price"))
            if not symbol or price is None or price <= 0:
                continue
            records.append(
                {
                    "name": entry.get("name") or symbol,
                    "symbol": symbol,
                    "price": price,
                    "change_24h": _to_float(entry.get("price_change_percentage_24h")),
                    "market_cap": _to_float(entry.get("market_cap")),
                    "source": "api",
                }
            )
        if not records:
            raise FallbackError("CoinGecko returned no usable rows")
        return records


class CoinMarketCapProvider(_Provider):
    """Second fallback: CoinMarketCap Pro API, only when an API key is set."""

    name = "coinmarketcap"

    def api_key(self) -> str:
        import os

        return os.environ.get(str(self.cfg.get("fallback.api_key_env", "CMC_API_KEY")), "").strip()

    def available(self) -> bool:
        return bool(self.api_key())

    def fetch_top(self, top_n: int) -> list[dict[str, Any]]:
        from scraper import extract_listings_from_payload  # shared, layout-agnostic parser

        base = str(self.cfg.get("fallback.coinmarketcap_base_url", "https://pro-api.coinmarketcap.com/v1")).rstrip("/")
        payload = self._get(
            f"{base}/cryptocurrency/listings/latest",
            params={
                "start": 1,
                "limit": max(1, int(top_n)),
                "convert": self.cfg.get("fallback.vs_currency", "usd"),
            },
            headers={"X-CMC_PRO_API_KEY": self.api_key(), "Accept": "application/json"},
        )
        records = extract_listings_from_payload(payload, top_n=top_n)
        for record in records:
            record["source"] = "api"
        if not records:
            raise FallbackError("CoinMarketCap returned no usable rows")
        return records


PROVIDERS = {"coingecko": CoinGeckoProvider, "coinmarketcap": CoinMarketCapProvider}


class FallbackFetcher:
    """Try each configured provider in order until one produces a batch."""

    def __init__(
        self,
        cfg: Any,
        session: requests.Session | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.cfg = cfg
        self.log = logger or LOGGER
        configured = cfg.get("fallback.providers", ["coingecko"]) or ["coingecko"]
        self.providers: list[_Provider] = []
        for name in configured:
            factory = PROVIDERS.get(str(name).lower())
            if factory is None:
                self.log.warning("unknown fallback provider configured: %s", name)
                continue
            self.providers.append(factory(cfg, session=session, logger=self.log))

    def fetch_top(self, top_n: int) -> tuple[list[dict[str, Any]], str]:
        """Return ``(records, provider_name)``; raise if every provider fails."""
        errors: list[str] = []
        for provider in self.providers:
            if not provider.available():
                self.log.info("fallback provider %s skipped (not configured)", provider.name)
                continue
            try:
                records = provider.fetch_top(top_n)
                self.log.info("fallback provider %s returned %d rows", provider.name, len(records))
                return records, provider.name
            except Exception as exc:
                self.log.warning("fallback provider %s failed: %s", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise FallbackError("all fallback providers failed -> " + ("; ".join(errors) or "none configured"))


def sample_records(top_n: int = 10, seed: int = 7) -> list[dict[str, Any]]:
    """Deterministic offline batch, tagged ``source="sample"``.

    Used by ``--source sample`` so the pipeline, alerts and dashboard can be
    demoed (or tested) without Chrome and without network access.
    """
    rng = random.Random(seed)
    universe = [
        ("Bitcoin", "BTC", 117_421.32, 2_340_000_000_000.0),
        ("Ethereum", "ETH", 4_312.55, 520_000_000_000.0),
        ("Tether", "USDT", 1.0, 168_000_000_000.0),
        ("BNB", "BNB", 1_012.44, 147_000_000_000.0),
        ("Solana", "SOL", 236.18, 128_000_000_000.0),
        ("XRP", "XRP", 2.87, 165_000_000_000.0),
        ("USD Coin", "USDC", 1.0, 74_000_000_000.0),
        ("Dogecoin", "DOGE", 0.2413, 36_000_000_000.0),
        ("Cardano", "ADA", 0.9124, 32_000_000_000.0),
        ("Tron", "TRX", 0.2861, 24_000_000_000.0),
    ]
    records: list[dict[str, Any]] = []
    for name, symbol, price, market_cap in universe[: max(1, int(top_n))]:
        drift = rng.uniform(-0.06, 0.06)
        records.append(
            {
                "name": name,
                "symbol": symbol,
                "price": round(price * (1 + drift), 8),
                "change_24h": round(drift * 100, 2),
                "market_cap": market_cap * (1 + drift),
                "source": "sample",
            }
        )
    return records
