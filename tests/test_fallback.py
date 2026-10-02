"""Stage 4 tests: fallback providers, ordering and the offline sample source."""

import pytest

from fallback import (
    CoinGeckoProvider,
    FallbackError,
    FallbackFetcher,
    sample_records,
)

COINGECKO_PAYLOAD = [
    {
        "name": "Bitcoin",
        "symbol": "btc",
        "current_price": 117_421.32,
        "price_change_percentage_24h": 1.24,
        "market_cap": 2.34e12,
    },
    {
        "name": "Ethereum",
        "symbol": "eth",
        "current_price": 4312.55,
        "price_change_percentage_24h": -0.5,
        "market_cap": 5.2e11,
    },
    {"name": "Broken", "symbol": "BAD", "current_price": None, "market_cap": None},
]

CMC_PAYLOAD = {
    "data": [
        {
            "name": "Bitcoin",
            "symbol": "BTC",
            "quote": {"USD": {"price": 117_500.0, "percent_change_24h": 1.3, "market_cap": 2.35e12}},
        }
    ]
}


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")

    def json(self):
        return self.payload


class FakeSession:
    """Records requests and replays canned payloads per URL fragment."""

    def __init__(self, routes):
        self.routes = routes
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append({"url": url, **kwargs})
        for fragment, payload in self.routes.items():
            if fragment in url:
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(payload)
        return FakeResponse({}, status=404)


class TestCoinGeckoProvider:
    def test_maps_fields_and_tags_source(self, cfg):
        session = FakeSession({"coins/markets": COINGECKO_PAYLOAD})
        records = CoinGeckoProvider(cfg, session=session).fetch_top(10)
        assert [r["symbol"] for r in records] == ["BTC", "ETH"]  # broken row skipped
        assert all(r["source"] == "api" for r in records)
        assert records[0]["price"] == pytest.approx(117_421.32)

    def test_passes_expected_query_parameters(self, cfg):
        session = FakeSession({"coins/markets": COINGECKO_PAYLOAD})
        CoinGeckoProvider(cfg, session=session).fetch_top(10)
        params = session.calls[0]["params"]
        assert params["vs_currency"] == "usd"
        assert params["order"] == "market_cap_desc"
        assert params["per_page"] == 10

    def test_empty_payload_raises(self, cfg):
        session = FakeSession({"coins/markets": []})
        with pytest.raises(FallbackError):
            CoinGeckoProvider(cfg, session=session).fetch_top(10)


class TestFallbackFetcher:
    def test_uses_coingecko_first(self, cfg):
        session = FakeSession({"coins/markets": COINGECKO_PAYLOAD})
        records, provider = FallbackFetcher(cfg, session=session).fetch_top(10)
        assert provider == "coingecko"
        assert records

    def test_falls_through_to_coinmarketcap(self, cfg, monkeypatch):
        monkeypatch.setenv("CMC_API_KEY", "test-key")
        session = FakeSession({"coins/markets": RuntimeError("rate limited"), "listings/latest": CMC_PAYLOAD})
        records, provider = FallbackFetcher(cfg, session=session).fetch_top(10)
        assert provider == "coinmarketcap"
        assert records[0]["source"] == "api"
        assert records[0]["price"] == pytest.approx(117_500.0)

    def test_raises_when_every_provider_fails(self, cfg, monkeypatch):
        monkeypatch.delenv("CMC_API_KEY", raising=False)
        session = FakeSession({"coins/markets": RuntimeError("down")})
        with pytest.raises(FallbackError):
            FallbackFetcher(cfg, session=session).fetch_top(10)

    def test_unknown_provider_is_skipped(self, cfg):
        fetcher = FallbackFetcher(cfg.with_overrides({"fallback.providers": ["nonsense", "coingecko"]}), session=FakeSession({}))
        assert [p.name for p in fetcher.providers] == ["coingecko"]


class TestSampleSource:
    def test_returns_requested_size(self):
        records = sample_records(10)
        assert len(records) == 10
        assert all(r["source"] == "sample" and r["price"] > 0 for r in records)
        assert records[0]["symbol"] == "BTC"

    def test_is_deterministic(self):
        assert sample_records(5, seed=1) == sample_records(5, seed=1)
        assert sample_records(5, seed=1) != sample_records(5, seed=2)
