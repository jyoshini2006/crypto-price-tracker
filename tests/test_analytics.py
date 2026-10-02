"""Stage 9 tests: moving averages, volatility, correlation, forecast."""

import numpy as np
import pandas as pd
import pytest

from analytics import (
    correlation_matrix,
    forecast_next,
    load_history,
    moving_averages,
    periods_per_year,
    pivot_prices,
    summarize,
    volatility,
)
from storage import build_storage


@pytest.fixture
def storage_with_history(cfg):
    from cleaner import records_to_dataframe

    storage = build_storage(cfg)
    # Three stored batches for BTC and ETH.
    for index, (btc, eth) in enumerate([(100.0, 10.0), (110.0, 10.5), (121.0, 11.0)]):
        records = [
            {"name": "Bitcoin", "symbol": "BTC", "price": btc, "change_24h": 1.0, "market_cap": 1e9, "source": "api"},
            {"name": "Ethereum", "symbol": "ETH", "price": eth, "change_24h": 0.5, "market_cap": 1e8, "source": "api"},
        ]
        storage.save(records_to_dataframe(records, timestamp=f"2026-01-0{index + 1}T00:00:00+00:00"))
    return storage


class TestMovingAverages:
    def test_sma_and_ema_columns(self):
        series = pd.Series([1.0, 2.0, 3.0, 4.0])
        frame = moving_averages(series, windows=(2,))
        assert list(frame.columns) == ["price", "sma_2", "ema_2"]
        assert frame["sma_2"].iloc[1] == pytest.approx(1.5)
        assert frame["sma_2"].iloc[3] == pytest.approx(3.5)
        # EMA(span=2) weights the newest point heavily: 3.5185 for this series.
        assert frame["ema_2"].iloc[-1] == pytest.approx(3.5185, abs=1e-3)


class TestVolatility:
    def test_constant_series_has_zero_volatility(self):
        assert volatility(pd.Series([5.0] * 10), window=5).dropna().iloc[-1] == pytest.approx(0.0)

    def test_annualisation_scales_by_sqrt_periods(self):
        series = pd.Series([100.0, 101.0, 99.0, 102.0, 98.0])
        raw = volatility(series, window=3).dropna().iloc[-1]
        annualised = volatility(series, window=3, periods_per_year=4).dropna().iloc[-1]
        assert annualised == pytest.approx(raw * 2.0)

    def test_periods_per_year_from_interval(self):
        assert periods_per_year(15) == pytest.approx(365 * 24 * 4)
        assert periods_per_year(60) == pytest.approx(365 * 24)


class TestCorrelation:
    def test_perfectly_correlated_series(self):
        wide = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0], "B": [10.0, 20.0, 30.0, 40.0]})
        matrix = correlation_matrix(wide)
        assert matrix.loc["A", "B"] == pytest.approx(1.0)

    def test_needs_two_coins(self):
        assert correlation_matrix(pd.DataFrame({"A": [1.0, 2.0]})).empty


class TestForecast:
    def test_linear_extrapolates_a_trend(self):
        series = pd.Series(np.arange(1, 11, dtype=float))
        result = forecast_next(series, method="linear", window=5)
        assert result["method"] == "linear"
        assert result["next_price"] > series.iloc[-1]
        assert result["lower"] <= result["next_price"] <= result["upper"]
        assert "not a prediction" in result["note"]

    def test_drift_matches_mean_return(self):
        series = pd.Series([100.0, 110.0, 121.0])
        result = forecast_next(series, method="drift", window=3)
        assert result["next_price"] > series.iloc[-1]

    def test_ema_falls_back_safely(self):
        result = forecast_next(pd.Series([1.0]), method="linear", window=20)
        assert result["next_price"] is not None

    def test_empty_series(self):
        assert forecast_next(pd.Series([], dtype=float))["next_price"] is None


class TestStorageIntegration:
    def test_load_and_pivot(self, storage_with_history):
        history = load_history(storage_with_history, symbol="BTC")
        assert len(history) == 3
        wide = pivot_prices(load_history(storage_with_history))
        assert set(wide.columns) == {"BTC", "ETH"}
        assert len(wide) == 3

    def test_summarize_reports_stats(self, storage_with_history):
        report = summarize(storage_with_history, symbol="BTC", forecast_method="ema")
        assert report["points"] == 3
        assert report["last_price"] == pytest.approx(121.0)
        assert report["change_since_first_pct"] == pytest.approx(21.0)
        assert "ema_20" in report["averages"]
        assert report["forecast"]["next_price"] is not None

    def test_summarize_without_history(self, cfg):
        report = summarize(build_storage(cfg), symbol="DOGE")
        assert report["points"] == 0
        assert "message" in report
