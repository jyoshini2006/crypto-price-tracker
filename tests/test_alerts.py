"""Stage 8 tests: threshold, spike and mover rules plus dry-run delivery."""

import pandas as pd
import pytest

from alerts import AlertEngine
from cleaner import records_to_dataframe
from storage import build_storage


@pytest.fixture
def engine(cfg):
    return AlertEngine(cfg, storage=build_storage(cfg))


def make_frame(rows):
    records = [
        {"name": name, "symbol": symbol, "price": price, "change_24h": change, "market_cap": 1e9, "source": "api"}
        for name, symbol, price, change in rows
    ]
    return records_to_dataframe(records)


class TestThresholds:
    def test_fires_when_crossing_up(self, engine):
        frame = make_frame([("Bitcoin", "BTC", 100_500.0, 1.0)])
        alerts = engine.evaluate(frame, previous_prices={"BTC": 99_000.0})
        assert any(a.kind == "threshold" and a.symbol == "BTC" for a in alerts)

    def test_fires_on_first_observation(self, engine):
        frame = make_frame([("Bitcoin", "BTC", 101_000.0, 1.0)])
        alerts = engine.evaluate(frame, previous_prices={})
        assert any(a.kind == "threshold" for a in alerts)

    def test_silent_when_already_above(self, engine):
        frame = make_frame([("Bitcoin", "BTC", 101_000.0, 1.0)])
        alerts = engine.evaluate(frame, previous_prices={"BTC": 100_500.0})
        assert not any(a.kind == "threshold" for a in alerts)

    def test_silent_below_threshold(self, engine):
        frame = make_frame([("Bitcoin", "BTC", 90_000.0, 1.0)])
        assert not any(a.kind == "threshold" for a in engine.evaluate(frame))


class TestSpikes:
    def test_spike_up_and_down(self, engine):
        frame = make_frame([("Solana", "SOL", 210.0, 6.0), ("Cardano", "ADA", 0.80, -7.0)])
        alerts = engine.evaluate(frame, previous_prices={"SOL": 200.0, "ADA": 1.0})
        spikes = [a for a in alerts if a.kind == "spike"]
        assert {a.symbol for a in spikes} == {"SOL", "ADA"}

    def test_small_move_is_ignored(self, engine):
        frame = make_frame([("Solana", "SOL", 202.0, 1.0)])
        alerts = engine.evaluate(frame, previous_prices={"SOL": 200.0})
        assert not any(a.kind == "spike" for a in alerts)

    def test_no_previous_price_means_no_spike(self, engine):
        frame = make_frame([("Solana", "SOL", 500.0, 1.0)])
        assert not any(a.kind == "spike" for a in engine.evaluate(frame, previous_prices={}))


class TestMovers:
    def test_reports_gainers_and_losers(self, engine):
        frame = make_frame(
            [
                ("A", "AAA", 1.0, 8.0),
                ("B", "BBB", 1.0, 5.0),
                ("C", "CCC", 1.0, -9.0),
                ("D", "DDD", 1.0, -1.0),
            ]
        )
        alerts = [a for a in engine.evaluate(frame) if a.kind == "movers"]
        assert len(alerts) == 2
        messages = " | ".join(a.message for a in alerts)
        assert "AAA" in messages and "CCC" in messages
        # top_movers=3 caps each list at 3
        assert len(alerts[0].meta["symbols"]) <= 3


class TestDelivery:
    def test_dry_run_does_not_deliver_but_is_persisted(self, engine, sample_batch):
        engine.dry_run = True
        frame = make_frame([("Bitcoin", "BTC", 101_000.0, 1.0)])
        alerts = engine.run(frame, previous_prices={})
        assert alerts
        assert all(not a.delivered for a in alerts)

        stored = engine.storage.recent_alerts()
        assert not stored.empty
        assert "BTC" in " ".join(stored["message"])

    def test_alert_log_file_written(self, engine, sample_batch):
        frame = make_frame([("Bitcoin", "BTC", 101_000.0, 1.0)])
        engine.run(frame, previous_prices={})
        assert engine.alert_file.exists()
        assert "threshold" in engine.alert_file.read_text(encoding="utf-8")

    def test_telegram_channel_called(self, engine, sample_batch):
        calls = {}

        class FakeResponse:
            def raise_for_status(self):
                return None

        class FakeSession:
            def post(self, url, json=None, timeout=None):
                calls["url"] = url
                calls["json"] = json
                return FakeResponse()

        engine.session = FakeSession()
        engine.dry_run = False
        engine.channels = {
            "telegram": {
                "enabled": True,
                "bot_token_env": "T",  # patched below via monkeypatch of os.environ
                "chat_id_env": "C",
            }
        }
        import os

        os.environ["T"] = "123:abc"
        os.environ["C"] = "42"
        try:
            frame = make_frame([("Bitcoin", "BTC", 101_000.0, 1.0)])
            alerts = engine.run(frame, previous_prices={})
        finally:
            del os.environ["T"]
            del os.environ["C"]

        assert calls["url"].endswith("/bot123:abc/sendMessage")
        assert "BTC" in calls["json"]["text"]
        assert all(a.delivered and a.channel == "telegram" for a in alerts)

    def test_disabled_engine_returns_nothing(self, engine, sample_batch):
        engine.enabled = False
        assert engine.run(sample_batch) == []

    def test_empty_frame_is_safe(self, engine):
        assert engine.evaluate(pd.DataFrame()) == []
