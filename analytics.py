"""Stage 9 - analytics over the stored history.

Everything here reads from SQLite through :class:`storage.Storage` and returns
plain pandas objects, so the same functions feed the Streamlit dashboard and the
command-line report::

    python analytics.py --symbol BTC --hours 168 --forecast
"""

from __future__ import annotations

import argparse
import logging
from typing import Any, Sequence

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

FORECAST_NOTE = (
    "experimental trend extrapolation over a few stored intervals - "
    "not investment advice and not a prediction"
)


def load_history(storage: Any, symbol: str | None = None, hours: int | None = None) -> pd.DataFrame:
    """History as a tidy frame with a real datetime index."""
    history = storage.history(symbol=symbol, hours=hours)
    if history.empty:
        return history
    history = history.copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True, errors="coerce")
    return history.dropna(subset=["timestamp"]).sort_values("timestamp")


def pivot_prices(history: pd.DataFrame) -> pd.DataFrame:
    """Wide price frame: one column per symbol, one row per timestamp."""
    if history.empty:
        return pd.DataFrame()
    wide = history.pivot_table(index="timestamp", columns="symbol", values="price", aggfunc="last")
    return wide.sort_index()


def moving_averages(series: pd.Series, windows: Sequence[int] = (5, 20)) -> pd.DataFrame:
    """Simple and exponential moving averages for a price series."""
    frame = pd.DataFrame({"price": series})
    for window in windows:
        window = int(window)
        if window < 1:
            continue
        frame[f"sma_{window}"] = series.rolling(window=window, min_periods=1).mean()
        frame[f"ema_{window}"] = series.ewm(span=window, adjust=False).mean()
    return frame


def volatility(
    series: pd.Series,
    window: int = 20,
    periods_per_year: float | None = None,
) -> pd.Series:
    """Rolling standard deviation of returns, optionally annualised."""
    returns = series.pct_change()
    rolling = returns.rolling(window=max(2, int(window)), min_periods=2).std()
    if periods_per_year:
        rolling = rolling * np.sqrt(float(periods_per_year))
    rolling.name = "volatility"
    return rolling


def periods_per_year(interval_minutes: float) -> float:
    """Convert a run interval into periods per year (for annualised volatility)."""
    minutes = max(1.0, float(interval_minutes))
    return 365.0 * 24.0 * 60.0 / minutes


def correlation_matrix(wide: pd.DataFrame, method: str = "pearson") -> pd.DataFrame:
    """Correlation of returns between coins."""
    if wide.empty or wide.shape[1] < 2:
        return pd.DataFrame()
    returns = wide.pct_change().dropna(how="all")
    if len(returns) < 2:
        return pd.DataFrame()
    return returns.corr(method=method, min_periods=2)


def forecast_next(series: pd.Series, method: str = "ema", window: int = 20) -> dict[str, Any]:
    """Very small next-interval estimator, clearly labelled experimental.

    ``ema``     - last exponential moving average value (smoothing, no drift),
    ``linear``  - least-squares line over the last ``window`` points,
    ``drift``   - last price plus the mean return over ``window`` points.
    """
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {"method": method, "next_price": None, "lower": None, "upper": None, "note": FORECAST_NOTE}

    window = max(2, int(window))
    recent = clean.tail(window)
    last = float(clean.iloc[-1])

    if method == "linear" and len(recent) >= 3:
        x = np.arange(len(recent), dtype=float)
        slope, intercept = np.polyfit(x, recent.to_numpy(dtype=float), 1)
        estimate = float(intercept + slope * len(recent))
    elif method == "drift" and len(recent) >= 2:
        drift = float(recent.pct_change().dropna().mean())
        estimate = last * (1.0 + drift)
    else:
        method = "ema"
        estimate = float(recent.ewm(span=window, adjust=False).mean().iloc[-1])

    sigma = float(recent.pct_change().dropna().std() or 0.0)
    band = 1.96 * sigma * last if sigma else 0.0
    return {
        "method": method,
        "last_price": last,
        "next_price": estimate,
        "lower": estimate - band,
        "upper": estimate + band,
        "sigma": sigma,
        "note": FORECAST_NOTE,
    }


def summarize(
    storage: Any,
    symbol: str,
    hours: int | None = None,
    windows: Sequence[int] = (5, 20),
    forecast_method: str = "ema",
    interval_minutes: float = 15.0,
) -> dict[str, Any]:
    """One-coin report used by the CLI and the dashboard."""
    history = load_history(storage, symbol=symbol, hours=hours)
    if history.empty:
        return {"symbol": symbol.upper(), "points": 0, "message": "no history stored yet"}

    series = history.set_index("timestamp")["price"]
    averages = moving_averages(series, windows)
    report: dict[str, Any] = {
        "symbol": symbol.upper(),
        "points": int(len(series)),
        "first_timestamp": series.index[0].isoformat(),
        "last_timestamp": series.index[-1].isoformat(),
        "last_price": float(series.iloc[-1]),
        "change_since_first_pct": float((series.iloc[-1] / series.iloc[0] - 1) * 100) if series.iloc[0] else None,
        "min_price": float(series.min()),
        "max_price": float(series.max()),
        "volatility": float(volatility(series, window=20, periods_per_year=periods_per_year(interval_minutes)).dropna().iloc[-1])
        if len(series) > 2
        else None,
        "averages": {column: float(averages[column].iloc[-1]) for column in averages.columns if column != "price"},
        "forecast": forecast_next(series, method=forecast_method, window=20),
    }
    return report


def _print_report(report: dict[str, Any]) -> None:
    if not report.get("points"):
        print(f"{report['symbol']}: {report.get('message', 'no data')}")
        return
    print(f"=== {report['symbol']} ===")
    print(f"points        : {report['points']} ({report['first_timestamp']} -> {report['last_timestamp']})")
    print(f"last price    : {report['last_price']:,.4f}")
    print(f"min / max     : {report['min_price']:,.4f} / {report['max_price']:,.4f}")
    if report.get("volatility") is not None:
        print(f"volatility    : {report['volatility']:.4f} (annualised)")
    for name, value in (report.get("averages") or {}).items():
        print(f"{name:<14}: {value:,.4f}")
    forecast = report.get("forecast") or {}
    if forecast.get("next_price") is not None:
        print(
            f"forecast({forecast['method']}): {forecast['next_price']:,.4f} "
            f"[{forecast['lower']:,.4f}, {forecast['upper']:,.4f}]"
        )
        print(f"note          : {forecast['note']}")


def main(argv: list[str] | None = None) -> int:
    from config import load_config
    from storage import build_storage

    parser = argparse.ArgumentParser(description="Analytics over the stored crypto history")
    parser.add_argument("--symbol", default="BTC", help="coin symbol (default: BTC)")
    parser.add_argument("--hours", type=int, default=None, help="history window in hours")
    parser.add_argument("--correlation", action="store_true", help="print the returns correlation matrix")
    parser.add_argument("--forecast", action="store_true", help="include the experimental forecast")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    storage = build_storage(cfg)

    interval = float(cfg.get("scheduler.interval_minutes", 15))
    report = summarize(
        storage,
        symbol=args.symbol,
        hours=args.hours,
        interval_minutes=interval,
        forecast_method="ema",
    )
    if not args.forecast:
        report.pop("forecast", None)
    _print_report(report)

    if args.correlation:
        wide = pivot_prices(load_history(storage, hours=args.hours))
        matrix = correlation_matrix(wide)
        print("\n=== correlation of returns ===")
        print(matrix.round(3).to_string() if not matrix.empty else "not enough history yet")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
