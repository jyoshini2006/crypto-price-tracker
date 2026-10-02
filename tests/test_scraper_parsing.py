"""Stage 3 tests: payload parsing, DOM heuristics and retry/backoff."""

from contextlib import contextmanager

import pytest
from selenium.common.exceptions import TimeoutException

from scraper import CoinMarketCapScraper, ScrapeError, extract_listings_from_payload


class TestEmbeddedJson:
    def test_next_data_layout(self):
        payload = {
            "props": {
                "pageProps": {
                    "initialState": {
                        "crypto": {
                            "listing": [
                                {
                                    "name": "Bitcoin",
                                    "symbol": "BTC",
                                    "price": 117_421.32,
                                    "percent_change_24h": 1.24,
                                    "marketCap": 2.34e12,
                                },
                                {
                                    "name": "Ethereum",
                                    "symbol": "ETH",
                                    "price": 4312.55,
                                    "percent_change_24h": -0.5,
                                    "marketCap": 5.2e11,
                                },
                            ]
                        }
                    }
                }
            }
        }
        records = extract_listings_from_payload(payload, top_n=10)
        assert [r["symbol"] for r in records] == ["BTC", "ETH"]
        assert records[0]["price"] == pytest.approx(117_421.32)
        assert records[0]["market_cap"] == pytest.approx(2.34e12)
        assert records[0]["source"] == "selenium"

    def test_coinmarketcap_api_layout(self):
        payload = {
            "status": {"error_code": 0},
            "data": [
                {
                    "name": "Bitcoin",
                    "symbol": "BTC",
                    "quote": {"USD": {"price": 117_421.32, "percent_change_24h": 1.24, "market_cap": 2.34e12}},
                }
            ],
        }
        records = extract_listings_from_payload(payload)
        assert len(records) == 1
        assert records[0]["price"] == pytest.approx(117_421.32)
        assert records[0]["change_24h"] == pytest.approx(1.24)
        assert records[0]["market_cap"] == pytest.approx(2.34e12)

    def test_string_values_with_currency_and_suffixes(self):
        payload = [{"symbol": "ETH", "price": "$4,312.55", "change_24h": "+2.1%", "market_cap": "$520.4B"}]
        record = extract_listings_from_payload(payload)[0]
        assert record["price"] == pytest.approx(4312.55)
        assert record["change_24h"] == pytest.approx(2.1)
        assert record["market_cap"] == pytest.approx(520.4e9)

    def test_deduplicates_and_limits(self):
        payload = [
            {"name": "Bitcoin", "symbol": "BTC", "price": 1.0},
            {"name": "Bitcoin again", "symbol": "BTC", "price": 2.0},
            {"name": "Ethereum", "symbol": "ETH", "price": 3.0},
        ]
        records = extract_listings_from_payload(payload, top_n=1)
        assert len(records) == 1
        assert records[0]["symbol"] == "BTC"

    def test_ignores_unrelated_objects(self):
        payload = {"user": {"symbol": "not a coin!", "name": "x"}, "empty": {}, "price": {"BTC": "n/a"}}
        assert extract_listings_from_payload(payload) == []

    def test_empty_payload(self):
        assert extract_listings_from_payload({}) == []
        assert extract_listings_from_payload(None) == []


class FakeElement:
    def __init__(self, text):
        self.text = text


class FakeRow:
    """Minimal stand-in for a Selenium WebElement table row."""

    def __init__(self, mapping=None, cells=()):
        self.mapping = mapping or {}
        self.cells = list(cells)

    def find_elements(self, by, selector):
        if selector == "td, [role='cell']":
            return [FakeElement(text) for text in self.cells]
        found = []
        for candidate in str(selector).split(","):
            candidate = candidate.strip()
            if candidate in self.mapping:
                found.append(FakeElement(self.mapping[candidate]))
        return found


class TestDomFallback:
    def test_uses_configured_selectors(self, cfg):
        scraper = CoinMarketCapScraper(cfg)
        # The config selector lists comma-separated alternatives; the fake row
        # answers for the first candidate of each one.
        row = FakeRow(
            {
                "p[class*='coin-item-name']": "Bitcoin BTC",
                "[data-testid='price']": "$117,421.32",
                "[data-testid='percent-change-24h']": "+1.24%",
                "[data-testid='market-cap']": "$2.34T",
            }
        )
        record = scraper._row_from_dom(row)
        assert record["symbol"] == "BTC"
        assert record["name"] == "Bitcoin"
        assert record["price"] == pytest.approx(117_421.32)
        assert record["market_cap"] == pytest.approx(2.34e12)

    def test_falls_back_to_cell_heuristics(self, cfg):
        scraper = CoinMarketCapScraper(cfg)
        row = FakeRow(cells=["1", "Bitcoin", "BTC", "$117,421.32", "+1.24%", "$2.34T"])
        record = scraper._row_from_dom(row)
        assert record["symbol"] == "BTC"
        assert record["price"] == pytest.approx(117_421.32)
        assert record["change_24h"] == pytest.approx(1.24)
        assert record["market_cap"] == pytest.approx(2.34e12)

    def test_unparseable_row_is_skipped(self, cfg):
        scraper = CoinMarketCapScraper(cfg)
        assert scraper._row_from_dom(FakeRow(cells=["1", "Bitcoin", "BTC", "n/a"])) is None


class TestRetries:
    @pytest.fixture
    def scraper(self, cfg, monkeypatch):
        instance = CoinMarketCapScraper(cfg.with_overrides({"scraper.retries": 3, "scraper.backoff_base": 0.0}))
        monkeypatch.setattr("scraper.time.sleep", lambda *_: None)  # no real waiting

        @contextmanager
        def fake_driver(self):
            yield object()

        monkeypatch.setattr(CoinMarketCapScraper, "driver", fake_driver)
        monkeypatch.setattr(CoinMarketCapScraper, "_load", lambda self, driver: None)
        return instance

    def test_retries_then_succeeds(self, scraper, monkeypatch):
        attempts = {"count": 0}

        def flaky(self, driver):
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise TimeoutException("table never appeared")
            return [{"name": "Bitcoin", "symbol": "BTC", "price": 1.0, "source": "selenium"}]

        monkeypatch.setattr(CoinMarketCapScraper, "_extract", flaky)
        records = scraper.fetch_top()
        assert attempts["count"] == 3
        assert records[0]["symbol"] == "BTC"

    def test_gives_up_with_scrape_error(self, scraper, monkeypatch):
        attempts = {"count": 0}

        def always_fails(self, driver):
            attempts["count"] += 1
            raise TimeoutException("still nothing")

        monkeypatch.setattr(CoinMarketCapScraper, "_extract", always_fails)
        with pytest.raises(ScrapeError):
            scraper.fetch_top()
        assert attempts["count"] == 3
