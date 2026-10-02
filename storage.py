"""Stage 6 - storage: append-only CSV plus a queryable SQLite history.

* CSV keeps the raw append-only log the assignment asks for (header written
  only when the file is created) and rotates into ``data/archive/prices_YYYY-MM.csv``
  at the first run of a new month.
* SQLite is the query layer: ``prices`` for history, ``runs`` for scraper
  health, ``alerts`` for the alert log the dashboard renders.
"""

from __future__ import annotations

import logging
import shutil
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import pandas as pd

from cleaner import COLUMNS, utc_timestamp

LOGGER = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS prices (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    coin       TEXT    NOT NULL,
    symbol     TEXT    NOT NULL,
    price      REAL    NOT NULL,
    change_24h REAL,
    market_cap REAL,
    timestamp  TEXT    NOT NULL,
    source     TEXT    NOT NULL,
    UNIQUE (symbol, timestamp)
);
CREATE INDEX IF NOT EXISTS idx_prices_symbol_ts ON prices (symbol, timestamp);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT,
    finished_at TEXT,
    status      TEXT NOT NULL,
    rows        INTEGER DEFAULT 0,
    source      TEXT,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS alerts (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    kind       TEXT NOT NULL,
    symbol     TEXT,
    message    TEXT NOT NULL,
    severity   TEXT DEFAULT 'info',
    delivered  INTEGER DEFAULT 0,
    channel    TEXT
);
"""


class Storage:
    """CSV + SQLite writer and query facade."""

    def __init__(
        self,
        csv_path: str | Path,
        sqlite_path: str | Path,
        archive_dir: str | Path | None = None,
        rotate_monthly: bool = True,
        logger: logging.Logger | None = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.sqlite_path = Path(sqlite_path)
        self.archive_dir = Path(archive_dir) if archive_dir else self.csv_path.parent / "archive"
        self.rotate_monthly = rotate_monthly
        self.log = logger or LOGGER
        self._prepare_directories()
        self.init_db()

    # -- setup ------------------------------------------------------------
    def _prepare_directories(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        if self.rotate_monthly:
            self.archive_dir.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.sqlite_path, timeout=30)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
            connection.commit()
        finally:
            connection.close()

    def init_db(self) -> None:
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    # -- CSV --------------------------------------------------------------
    def rotate_csv_if_needed(self, moment: datetime | None = None) -> Path | None:
        """Archive the current CSV when it belongs to an earlier month."""
        if not self.rotate_monthly or not self.csv_path.exists():
            return None

        moment = moment or datetime.now(timezone.utc)
        modified = datetime.fromtimestamp(self.csv_path.stat().st_mtime, tz=timezone.utc)
        if (modified.year, modified.month) == (moment.year, moment.month):
            return None

        archived = self.archive_dir / f"prices_{modified.year:04d}-{modified.month:02d}.csv"
        counter = 1
        while archived.exists():
            archived = self.archive_dir / f"prices_{modified.year:04d}-{modified.month:02d}_{counter}.csv"
            counter += 1
        shutil.move(str(self.csv_path), str(archived))
        self.log.info("rotated CSV to %s", archived)
        return archived

    def append_csv(self, frame: pd.DataFrame) -> Path:
        """Append a batch to the CSV, writing the header only for a new file."""
        self.rotate_csv_if_needed()
        is_new = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        frame.to_csv(
            self.csv_path,
            mode="a",
            header=is_new,
            index=False,
            columns=list(COLUMNS),
            encoding="utf-8",
        )
        return self.csv_path

    # -- SQLite writes ----------------------------------------------------
    def insert_prices(self, frame: pd.DataFrame) -> int:
        """Insert a batch, ignoring rows already stored for the same timestamp."""
        rows = [
            (
                row.coin,
                row.symbol,
                float(row.price),
                None if pd.isna(row.change_24h) else float(row.change_24h),
                None if pd.isna(row.market_cap) else float(row.market_cap),
                row.timestamp,
                row.source,
            )
            for row in frame.itertuples(index=False)
        ]
        with self._connect() as connection:
            before = connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
            connection.executemany(
                """INSERT OR IGNORE INTO prices
                   (coin, symbol, price, change_24h, market_cap, timestamp, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            after = connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        return after - before

    def save(self, frame: pd.DataFrame) -> dict[str, Any]:
        """Persist a validated batch to both sinks."""
        csv_path = self.append_csv(frame)
        inserted = self.insert_prices(frame)
        return {"csv_path": csv_path, "inserted": inserted, "rows": len(frame)}

    # -- run / alert logging ---------------------------------------------
    def record_run(
        self,
        status: str,
        rows: int = 0,
        source: str | None = None,
        error: str | None = None,
        started_at: str | None = None,
        finished_at: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO runs (started_at, finished_at, status, rows, source, error)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    started_at or utc_timestamp(),
                    finished_at or utc_timestamp(),
                    status,
                    int(rows),
                    source,
                    error,
                ),
            )
            return int(cursor.lastrowid or 0)

    def log_alert(
        self,
        kind: str,
        message: str,
        symbol: str | None = None,
        severity: str = "info",
        delivered: bool = False,
        channel: str | None = None,
        created_at: str | None = None,
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO alerts (created_at, kind, symbol, message, severity, delivered, channel)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    created_at or utc_timestamp(),
                    kind,
                    symbol,
                    message,
                    severity,
                    1 if delivered else 0,
                    channel,
                ),
            )
            return int(cursor.lastrowid or 0)

    # -- reads ------------------------------------------------------------
    def _query(self, sql: str, params: Iterable[Any] = ()) -> pd.DataFrame:
        with self._connect() as connection:
            return pd.read_sql_query(sql, connection, params=tuple(params))

    def price_count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM prices").fetchone()[0])

    def latest_snapshot(self) -> pd.DataFrame:
        """Most recent batch, one row per coin."""
        return self._query(
            """SELECT coin, symbol, price, change_24h, market_cap, timestamp, source
               FROM prices
               WHERE timestamp = (SELECT MAX(timestamp) FROM prices)
               ORDER BY market_cap IS NULL, market_cap DESC, price DESC"""
        )

    def history(self, symbol: str | None = None, hours: int | None = None) -> pd.DataFrame:
        """Price history for one coin (or all coins), optionally windowed."""
        clauses: list[str] = []
        params: list[Any] = []
        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol.upper())
        if hours:
            cutoff = utc_timestamp(datetime.now(timezone.utc) - timedelta(hours=hours))
            clauses.append("timestamp >= ?")
            params.append(cutoff)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        return self._query(
            f"""SELECT symbol, coin, price, change_24h, market_cap, timestamp, source
                FROM prices {where} ORDER BY timestamp ASC, symbol ASC""",
            params,
        )

    def previous_prices(self, symbols: Iterable[str]) -> dict[str, float]:
        """Last stored price per symbol, used for spike detection.

        Call this *before* saving the new batch so "previous" really means
        previous.
        """
        result: dict[str, float] = {}
        with self._connect() as connection:
            for symbol in symbols:
                row = connection.execute(
                    """SELECT price FROM prices WHERE symbol = ?
                       ORDER BY timestamp DESC, id DESC LIMIT 1""",
                    (str(symbol).upper(),),
                ).fetchone()
                if row is not None:
                    result[str(symbol).upper()] = float(row["price"])
        return result

    def recent_runs(self, limit: int = 20) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM runs ORDER BY id DESC LIMIT ?", (int(limit),)
        )

    def recent_alerts(self, limit: int = 50) -> pd.DataFrame:
        return self._query(
            "SELECT * FROM alerts ORDER BY id DESC LIMIT ?", (int(limit),)
        )

    def health_summary(self) -> dict[str, Any]:
        """Small health snapshot for the dashboard / CLI status line."""
        runs = self.recent_runs(limit=20)
        if runs.empty:
            return {"runs": 0, "last_status": None, "last_source": None, "fallback_runs": 0, "rows": self.price_count()}
        last = runs.iloc[0]
        return {
            "runs": int(len(runs)),
            "last_status": last["status"],
            "last_source": last["source"],
            "last_finished_at": last["finished_at"],
            "last_rows": int(last["rows"] or 0),
            "fallback_runs": int((runs["source"] == "api").sum()),
            "failed_runs": int(runs["status"].isin(["failed", "rejected"]).sum()),
            "rows": self.price_count(),
        }


def build_storage(cfg: Any, logger: logging.Logger | None = None) -> Storage:
    """Create a :class:`Storage` from a :class:`config.Config`."""
    return Storage(
        csv_path=cfg.path("storage.csv_path"),
        sqlite_path=cfg.path("storage.sqlite_path"),
        archive_dir=cfg.path("storage.archive_dir"),
        rotate_monthly=bool(cfg.get("storage.rotate_monthly", True)),
        logger=logger,
    )
