"""Stage 5 tests: parsing and validation."""

import pytest

from cleaner import (
    BadBatchError,
    clean_and_validate,
    parse_market_cap,
    parse_number,
    parse_percent,
    parse_price,
    parse_symbol,
    records_to_dataframe,
    utc_timestamp,
    validate_batch,
)


class TestParseNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$117,421.32", 117_421.32),
            ("1,234.56", 1234.56),
            ("$0.00001234", 0.00001234),
            ("+1.24%", 1.24),
            ("-3.5%", -3.5),
            ("\u22121.02%", -1.02),  # unicode minus
            ("$2.34T", 2.34e12),
            ("1.5B", 1.5e9),
            ("980.2M", 980.2e6),
            ("12K", 12_000.0),
            ("(1,234.5)", -1234.5),
            (12.5, 12.5),
            (0, 0.0),
        ],
    )
    def test_parses_common_formats(self, raw, expected):
        assert parse_number(raw) == pytest.approx(expected)

    @pytest.mark.parametrize("raw", [None, "", "  ", "-", "--", "N/A", "n/a", "—", "?", "abc", "K", "$", float("nan")])
    def test_returns_none_for_unusable_values(self, raw):
        assert parse_number(raw) is None

    def test_rejects_booleans(self):
        assert parse_number(True) is None

    def test_helpers_delegate_to_parse_number(self):
        assert parse_price("$42,000.50") == pytest.approx(42_000.50)
        assert parse_percent("-8.25%") == pytest.approx(-8.25)
        assert parse_market_cap("$1.2T") == pytest.approx(1.2e12)

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("btc", "BTC"), (" eth ", "ETH"), ("$SOL", "SOL"), ("usdt", "USDT"), ("doge/usd", "DOGEUSD")],
    )
    def test_parse_symbol(self, raw, expected):
        assert parse_symbol(raw) == expected

    def test_parse_symbol_rejects_blank(self):
        assert parse_symbol("   ") is None


class TestRecordsToDataframe:
    def test_normalises_raw_source_strings(self):
        frame = records_to_dataframe(
            [{"name": "Bitcoin", "symbol": "btc", "price": "$117,421.32", "change_24h": "+1.24%", "market_cap": "$2.34T", "source": "Selenium"}]
        )
        row = frame.iloc[0]
        assert list(frame.columns) == ["coin", "symbol", "price", "change_24h", "market_cap", "timestamp", "source"]
        assert row["symbol"] == "BTC"
        assert row["price"] == pytest.approx(117_421.32)
        assert row["change_24h"] == pytest.approx(1.24)
        assert row["market_cap"] == pytest.approx(2.34e12)
        assert row["source"] == "selenium"
        assert row["timestamp"].endswith("+00:00")

    def test_accepts_numeric_api_records(self):
        frame = records_to_dataframe(
            [{"name": "Ethereum", "symbol": "ETH", "price": 4312.55, "change_24h": None, "market_cap": 5.2e11, "source": "api"}]
        )
        assert frame.iloc[0]["price"] == pytest.approx(4312.55)

    def test_timestamp_is_utc_iso(self):
        assert utc_timestamp().endswith("+00:00")


class TestValidateBatch:
    def test_accepts_a_clean_batch(self, sample_batch):
        report = validate_batch(sample_batch, expected_rows=10)
        assert report.ok, report.summary()
        assert report.rows == 10

    def test_rejects_wrong_row_count(self, sample_batch):
        report = validate_batch(sample_batch.head(9), expected_rows=10)
        assert not report.ok
        assert "expected 10 rows" in report.errors[0]

    def test_rejects_non_positive_price(self, sample_batch):
        broken = sample_batch.copy()
        broken.loc[0, "price"] = 0.0
        report = validate_batch(broken, expected_rows=10)
        assert not report.ok
        assert "non-positive" in " ".join(report.errors)

    def test_rejects_duplicate_coins(self, sample_batch):
        duplicated = sample_batch.copy()
        duplicated.loc[1, "symbol"] = duplicated.loc[0, "symbol"]
        report = validate_batch(duplicated, expected_rows=10)
        assert not report.ok
        assert "duplicate coins" in " ".join(report.errors)

    def test_rejects_empty_batch(self, sample_batch):
        report = validate_batch(sample_batch.head(0), expected_rows=10)
        assert not report.ok
        assert "empty" in " ".join(report.errors)

    def test_missing_market_cap_is_only_a_warning(self, sample_batch):
        frame = sample_batch.copy()
        frame.loc[0, "market_cap"] = None
        report = validate_batch(frame, expected_rows=10)
        assert report.ok
        assert "missing market cap" in " ".join(report.warnings)


class TestCleanAndValidate:
    def test_returns_frame_and_report(self):
        records = [
            {"name": f"Coin{i}", "symbol": f"C{i}", "price": 1.0 + i, "change_24h": 1.0, "market_cap": 1e9, "source": "api"}
            for i in range(10)
        ]
        frame, report = clean_and_validate(records, expected_rows=10)
        assert len(frame) == 10
        assert report.ok

    def test_raises_bad_batch_with_report(self):
        with pytest.raises(BadBatchError) as excinfo:
            clean_and_validate([{"symbol": "BTC", "price": 0, "source": "api"}], expected_rows=10)
        assert excinfo.value.report is not None
        assert not excinfo.value.report.ok

    def test_exact_row_count_is_optional(self):
        frame, report = clean_and_validate([{"symbol": "BTC", "price": 1.0, "source": "api"}], expected_rows=None)
        assert len(frame) == 1 and report.ok
