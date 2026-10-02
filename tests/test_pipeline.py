"""Stages 1/7/11 tests: the orchestrator, its statuses and its health records."""

import logging

import pytest

import main
from storage import build_storage


def run(cfg, **kwargs):
    log = logging.getLogger("test-pipeline")
    return main.run_once(cfg, log=log, **kwargs)


class TestHappyPath:
    def test_sample_source_writes_csv_and_sqlite(self, cfg):
        summary = run(cfg, source="sample")
        assert summary["status"] == "ok"
        assert summary["rows"] == 10

        storage = build_storage(cfg)
        assert storage.price_count() == 10
        csv_lines = storage.csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 11
        assert csv_lines[0].startswith("coin,symbol,price")

        runs = storage.recent_runs()
        assert runs.iloc[0]["status"] == "ok"
        assert runs.iloc[0]["source"] == "sample"
        assert int(runs.iloc[0]["rows"]) == 10

    def test_second_run_appends_without_duplicate_header(self, cfg):
        run(cfg, source="sample")
        summary = run(cfg, source="sample")
        assert summary["status"] == "ok"
        csv_lines = build_storage(cfg).csv_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(csv_lines) == 21
        assert sum(line.startswith("coin,symbol") for line in csv_lines) == 1

    def test_alerts_are_evaluated_and_logged(self, cfg):
        summary = run(cfg, source="sample")
        assert summary["alerts"] >= 1  # sample BTC is above the 100k threshold
        alerts = build_storage(cfg).recent_alerts()
        assert not alerts.empty
        assert (cfg.path("logging.dir") / str(cfg.get("logging.alert_file"))).exists()

    def test_alerts_can_be_disabled(self, cfg):
        summary = run(cfg.with_overrides({"alerts.enabled": False}), source="sample")
        assert summary["alerts"] == 0


class TestFallbackPath:
    def test_selenium_failure_uses_api_and_flags_degraded(self, cfg, monkeypatch):
        monkeypatch.setattr(main.CoinMarketCapScraper, "fetch_top", lambda self: (_ for _ in ()).throw(RuntimeError("no chrome")))
        monkeypatch.setattr(main.FallbackFetcher, "fetch_top", lambda self, n: (main.sample_records(n), "coingecko"))

        summary = run(cfg, source="auto")
        assert summary["status"] == "degraded"
        assert summary["used_source"] == "api"
        assert any("selenium failed" in note for note in summary["notes"])
        assert build_storage(cfg).health_summary()["fallback_runs"] == 1

    def test_source_selenium_does_not_fall_back(self, cfg, monkeypatch):
        monkeypatch.setattr(main.CoinMarketCapScraper, "fetch_top", lambda self: (_ for _ in ()).throw(RuntimeError("no chrome")))
        summary = run(cfg, source="selenium")
        assert summary["status"] == "failed"
        assert "selenium scraping failed" in summary["error"]
        assert build_storage(cfg).price_count() == 0


class TestFailureHandling:
    def test_all_sources_down_is_recorded(self, cfg, monkeypatch):
        monkeypatch.setattr(main.CoinMarketCapScraper, "fetch_top", lambda self: (_ for _ in ()).throw(RuntimeError("no chrome")))
        monkeypatch.setattr(main.FallbackFetcher, "fetch_top", lambda self, n: (_ for _ in ()).throw(RuntimeError("api down")))

        summary = run(cfg, source="auto")
        assert summary["status"] == "failed"
        runs = build_storage(cfg).recent_runs()
        assert runs.iloc[0]["status"] == "failed"
        assert "api down" in runs.iloc[0]["error"]

    def test_short_batch_is_rejected_not_stored(self, cfg, monkeypatch):
        monkeypatch.setattr(
            main,
            "acquire_records",
            lambda cfg, source, log: (
                [{"name": "Bitcoin", "symbol": "BTC", "price": 1.0, "change_24h": 1.0, "market_cap": 1e9, "source": "api"}],
                "api",
                [],
            ),
        )
        summary = run(cfg, source="auto")
        assert summary["status"] == "rejected"
        assert "got 1" in summary["error"]  # fewer than validation.min_rows
        storage = build_storage(cfg)
        assert storage.price_count() == 0
        assert storage.recent_runs().iloc[0]["status"] == "rejected"

    def test_zero_price_batch_is_rejected(self, cfg, monkeypatch):
        records = main.sample_records(10)
        records[3]["price"] = 0
        monkeypatch.setattr(main, "acquire_records", lambda cfg, source, log: (records, "sample", []))
        summary = run(cfg, source="sample")
        assert summary["status"] == "rejected"
        assert build_storage(cfg).price_count() == 0


class TestErrorTrimming:
    def test_multiline_driver_error_is_collapsed(self):
        raw = "Message: timeout\nStacktrace:\n\tchromedriver!GetHandleVerifier [0x1]\n"
        assert main.short_error(raw) == "Message: timeout"

    def test_long_errors_are_truncated(self):
        trimmed = main.short_error("x" * 1000)
        assert len(trimmed) == 300
        assert trimmed.endswith("\u2026")

    def test_runs_table_stores_one_liners(self, cfg, monkeypatch):
        monkeypatch.setattr(
            main.CoinMarketCapScraper,
            "fetch_top",
            lambda self: (_ for _ in ()).throw(RuntimeError("boom\nStacktrace:\n\tframe")),
        )
        run(cfg, source="selenium")
        error = build_storage(cfg).recent_runs().iloc[0]["error"]
        assert error == "selenium scraping failed: boom"


class TestLogging:
    def test_log_file_is_created_and_populated(self, cfg):
        log = main.setup_logging(cfg)
        try:
            # Use the configured logger so the rotating file handler sees it.
            main.run_once(cfg, source="sample", log=log)
        finally:
            for handler in log.handlers:
                handler.close()
        log_file = cfg.path("logging.dir") / str(cfg.get("logging.file"))
        assert log_file.exists()
        content = log_file.read_text(encoding="utf-8")
        assert "stored 10 rows" in content
        assert "run finished" in content

    def test_status_printer(self, cfg, capsys):
        run(cfg, source="sample")
        main.print_status(cfg, build_storage(cfg))
        output = capsys.readouterr().out
        assert "scraper health" in output
        assert "last_status" in output


class TestCli:
    def test_parser_defaults(self):
        args = main.build_parser().parse_args([])
        assert args.source == "auto"
        assert args.schedule is False

    def test_main_once_with_sample_source(self, cfg, monkeypatch, capsys, tmp_path):
        # Redirect the CLI's own config load at the temp paths.
        monkeypatch.setattr("config.load_config", lambda path=None: cfg)
        exit_code = main.main(["--source", "sample", "--once", "--dry-run"])
        assert exit_code == 0
        assert "status=ok" in capsys.readouterr().out

    def test_main_returns_nonzero_on_failure(self, cfg, monkeypatch, capsys):
        monkeypatch.setattr("config.load_config", lambda path=None: cfg)
        monkeypatch.setattr(main.CoinMarketCapScraper, "fetch_top", lambda self: (_ for _ in ()).throw(RuntimeError("boom")))
        monkeypatch.setattr(main.FallbackFetcher, "fetch_top", lambda self, n: (_ for _ in ()).throw(RuntimeError("boom")))
        assert main.main(["--once", "--no-alerts"]) == 1
        assert "status=failed" in capsys.readouterr().out
