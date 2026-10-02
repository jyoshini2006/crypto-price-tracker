"""Stage 6 tests: CSV append, monthly rotation, SQLite queries and health."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from storage import Storage


@pytest.fixture
def storage(tmp_path):
    return Storage(
        csv_path=tmp_path / "data" / "prices.csv",
        sqlite_path=tmp_path / "data" / "crypto.db",
        archive_dir=tmp_path / "data" / "archive",
    )


class TestCsv:
    def test_header_written_once(self, storage, sample_batch):
        storage.append_csv(sample_batch)
        storage.append_csv(sample_batch.assign(timestamp="2099-01-01T00:00:00+00:00"))
        lines = storage.csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert lines[0].startswith("coin,symbol,price")
        assert lines[0] not in lines[1:]
        assert len(lines) == 21  # 1 header + 20 data rows

    def test_rotation_archives_old_month(self, storage, sample_batch):
        storage.append_csv(sample_batch)
        # Pretend the file was written last month.
        last_month = datetime(2020, 1, 15, tzinfo=timezone.utc)
        import os

        stamp = last_month.timestamp()
        os.utime(storage.csv_path, (stamp, stamp))

        rotated = storage.rotate_csv_if_needed()
        assert rotated is not None and rotated.exists()
        assert rotated.name == "prices_2020-01.csv"
        assert not storage.csv_path.exists()  # next append starts a fresh file
        storage.append_csv(sample_batch)
        assert storage.csv_path.exists()

    def test_no_rotation_within_same_month(self, storage, sample_batch):
        storage.append_csv(sample_batch)
        assert storage.rotate_csv_if_needed() is None


class TestSqlite:
    def test_roundtrip(self, storage, sample_batch):
        storage.save(sample_batch)
        assert storage.price_count() == 10
        snapshot = storage.latest_snapshot()
        assert len(snapshot) == 10
        assert set(snapshot.columns) >= {"coin", "symbol", "price", "timestamp", "source"}

    def test_duplicate_timestamp_rows_are_ignored(self, storage, sample_batch):
        storage.save(sample_batch)
        inserted = storage.insert_prices(sample_batch)  # same timestamps again
        assert inserted == 0
        assert storage.price_count() == 10

    def test_history_and_previous_prices(self, storage, sample_batch):
        earlier = sample_batch.assign(timestamp="2020-01-01T00:00:00+00:00")
        later = sample_batch.assign(timestamp="2020-01-02T00:00:00+00:00", price=2.0)
        storage.save(earlier)
        storage.save(later)

        history = storage.history(symbol="BTC")
        assert len(history) == 2
        assert history.iloc[-1]["price"] == pytest.approx(2.0)  # ascending by time
        assert storage.history(hours=1).empty  # both batches are in 2020

        previous = storage.previous_prices(["BTC", "ETH", "NOPE"])
        assert previous["BTC"] == pytest.approx(2.0)  # the newest batch wins
        assert previous["ETH"] == pytest.approx(2.0)
        assert "NOPE" not in previous

    def test_runs_and_alerts_are_recorded(self, storage, sample_batch):
        storage.save(sample_batch)
        storage.record_run("ok", rows=10, source="selenium")
        storage.record_run("degraded", rows=10, source="api")
        storage.log_alert("threshold", "BTC above 100k", symbol="BTC", severity="warning")

        runs = storage.recent_runs()
        assert len(runs) == 2
        assert runs.iloc[0]["status"] == "degraded"

        health = storage.health_summary()
        assert health["rows"] == 10
        assert health["last_status"] == "degraded"
        assert health["last_source"] == "api"
        assert health["fallback_runs"] == 1
        assert health["failed_runs"] == 0

        alerts = storage.recent_alerts()
        assert alerts.iloc[0]["kind"] == "threshold"

    def test_empty_database_health(self, storage):
        health = storage.health_summary()
        assert health["rows"] == 0
        assert health["last_status"] is None

    def test_read_sql_returns_dataframe(self, storage, sample_batch):
        storage.save(sample_batch)
        frame = storage.history()
        assert isinstance(frame, pd.DataFrame)
        assert frame["price"].dtype.kind == "f"
