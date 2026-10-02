"""Stage 5 - cleaning and validation.

Raw rows arrive either from Selenium (strings like ``"$117,421.32"``,
``"2.34T"``, ``"+1.24%"``) or from an API (already numeric, sometimes with
``None``). Everything funnels through :func:`clean_and_validate`, which returns
a tidy DataFrame or raises :class:`BadBatchError` so a bad batch is logged and
rejected instead of being written to storage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping

import pandas as pd

#: Columns every stored record must have (order is also the CSV column order).
COLUMNS: tuple[str, ...] = ("coin", "symbol", "price", "change_24h", "market_cap", "timestamp", "source")

_NULL_TOKENS = {"", "-", "--", "n/a", "na", "nan", "none", "null", "—", "–", "?"}
_SUFFIXES = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
_STRIP_CHARS = "$\u20ac\u00a3\u00a5,%+ \u00a0\u2009'"


class BadBatchError(ValueError):
    """Raised when a batch fails validation and must not be stored."""

    def __init__(self, message: str, report: "ValidationReport | None" = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class ValidationReport:
    """Outcome of the sanity checks applied to one batch."""

    rows: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        parts = [f"rows={self.rows}", f"status={'ok' if self.ok else 'rejected'}"]
        if self.errors:
            parts.append("errors=[" + "; ".join(self.errors) + "]")
        if self.warnings:
            parts.append("warnings=[" + "; ".join(self.warnings) + "]")
        return " ".join(parts)


# ---------------------------------------------------------------------------
# primitive parsers
# ---------------------------------------------------------------------------
def parse_number(value: Any) -> float | None:
    """Parse ``"$1,234.56"``, ``"2.34T"``, ``"-1.2%"``, ``12.5`` into a float.

    Returns ``None`` for empty values, placeholder tokens and anything that is
    not a finite number, so callers can decide whether that is fatal.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None

    text = str(value).strip()
    if text.lower() in _NULL_TOKENS:
        return None

    negative = text.startswith("(") and text.endswith(")")
    text = text.replace("\u2212", "-").replace("(", "").replace(")", "").strip()

    multiplier = 1.0
    if text and text[-1].upper() in _SUFFIXES and not text[-1].isdigit():
        multiplier = _SUFFIXES[text[-1].upper()]
        text = text[:-1]

    # Thousands separators can appear anywhere, not just at the ends.
    text = text.strip().strip(_STRIP_CHARS).replace(" ", "").replace(",", "")
    if text in ("", "-", "+"):
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if not math.isfinite(number):
        return None
    if negative:
        number = -abs(number)
    return number * multiplier


def parse_price(value: Any) -> float | None:
    """Price parser (no suffix handling surprises: ``"$0.00001234"`` is a price)."""
    return parse_number(value)


def parse_percent(value: Any) -> float | None:
    """Parse ``"+1.24%"`` / ``"-3.5%"`` into ``1.24`` / ``-3.5``."""
    return parse_number(value)


def parse_market_cap(value: Any) -> float | None:
    """Parse market caps, expanding ``B``/``M``/``T`` suffixes to full numbers."""
    return parse_number(value)


def parse_symbol(value: Any) -> str | None:
    """Normalise a ticker to upper case, dropping punctuation and whitespace."""
    if value is None:
        return None
    symbol = str(value).strip().upper()
    symbol = symbol.split()[0] if symbol.split() else symbol
    symbol = "".join(ch for ch in symbol if ch.isalnum())
    return symbol or None


def parse_name(value: Any) -> str | None:
    if value is None:
        return None
    name = str(value).strip()
    return name or None


def utc_timestamp(moment: datetime | None = None) -> str:
    """ISO-8601 UTC timestamp used in both CSV and SQLite."""
    moment = moment or datetime.now(timezone.utc)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# batch pipeline
# ---------------------------------------------------------------------------
def records_to_dataframe(
    records: Iterable[Mapping[str, Any]],
    timestamp: str | None = None,
) -> pd.DataFrame:
    """Convert raw source records into the canonical column layout."""
    stamp = timestamp or utc_timestamp()
    rows: list[dict[str, Any]] = []
    for record in records:
        rows.append(
            {
                "coin": parse_name(record.get("name") or record.get("coin")),
                "symbol": parse_symbol(record.get("symbol")),
                "price": parse_price(record.get("price")),
                "change_24h": parse_percent(record.get("change_24h")),
                "market_cap": parse_market_cap(record.get("market_cap")),
                "timestamp": stamp,
                "source": str(record.get("source") or "unknown").strip().lower(),
            }
        )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["change_24h"] = pd.to_numeric(frame["change_24h"], errors="coerce")
    frame["market_cap"] = pd.to_numeric(frame["market_cap"], errors="coerce")
    return frame


def validate_batch(
    frame: pd.DataFrame,
    expected_rows: int | None = None,
    min_rows: int = 1,
) -> ValidationReport:
    """Run the sanity checks that decide whether a batch is trustworthy."""
    report = ValidationReport(rows=len(frame))

    missing = [column for column in COLUMNS if column not in frame.columns]
    if missing:
        report.errors.append(f"missing columns: {', '.join(missing)}")
        return report

    if frame.empty:
        report.errors.append("batch is empty")
        return report

    if expected_rows is not None and len(frame) != expected_rows:
        report.errors.append(f"expected {expected_rows} rows, got {len(frame)}")
    elif len(frame) < min_rows:
        report.errors.append(f"expected at least {min_rows} row(s), got {len(frame)}")

    prices = pd.to_numeric(frame["price"], errors="coerce")
    bad_prices = frame.loc[prices.isna() | (prices <= 0), "symbol"].astype(str).tolist()
    if bad_prices:
        report.errors.append(f"non-positive or unparseable price for: {', '.join(bad_prices)}")

    symbols = frame["symbol"].astype("string")
    if symbols.isna().any() or (symbols.str.len() == 0).any():
        report.errors.append("missing symbol")

    duplicates = sorted(set(symbols.dropna()[symbols.dropna().duplicated()].tolist()))
    if duplicates:
        report.errors.append(f"duplicate coins: {', '.join(duplicates)}")

    if frame["timestamp"].isna().any() or (frame["timestamp"].astype(str).str.len() == 0).any():
        report.errors.append("missing timestamp")

    market_caps = pd.to_numeric(frame["market_cap"], errors="coerce")
    unknown_caps = frame.loc[market_caps.isna() | (market_caps <= 0), "symbol"].astype(str).tolist()
    if unknown_caps:
        report.warnings.append(f"missing market cap for: {', '.join(unknown_caps)}")

    changes = pd.to_numeric(frame["change_24h"], errors="coerce")
    unknown_changes = frame.loc[changes.isna(), "symbol"].astype(str).tolist()
    if unknown_changes:
        report.warnings.append(f"missing 24h change for: {', '.join(unknown_changes)}")

    return report


def clean_and_validate(
    records: Iterable[Mapping[str, Any]],
    expected_rows: int | None = None,
    timestamp: str | None = None,
    min_rows: int = 1,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Clean raw records and validate them, raising on a bad batch.

    ``expected_rows`` is normally ``scraper.top_n`` (the spec's "exactly 10
    rows" rule). Pass ``None`` to accept a shorter batch, in which case
    ``min_rows`` still applies so a two-row API response is rejected too.
    """
    frame = records_to_dataframe(records, timestamp=timestamp)
    report = validate_batch(frame, expected_rows=expected_rows, min_rows=min_rows)
    if not report.ok:
        raise BadBatchError(f"batch rejected: {report.summary()}", report)
    return frame.reset_index(drop=True), report
