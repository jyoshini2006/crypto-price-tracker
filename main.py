"""Pipeline orchestrator - stages 1, 7 and 11.

    acquire (Selenium -> API fallback)  ->  clean & validate  ->  store
        ->  alerts  ->  log a health record

Run once, on a local schedule, or point a cron job / Task Scheduler at
``python main.py --once``::

    python main.py --once                     # one pass, Selenium then API
    python main.py --source sample --once     # offline demo, no Chrome needed
    python main.py --schedule                 # run every scheduler.interval_minutes
    python main.py --status                   # print scraper health
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from cleaner import BadBatchError, clean_and_validate, utc_timestamp
from fallback import FallbackFetcher, sample_records
from scraper import CoinMarketCapScraper
from storage import Storage, build_storage

LOGGER = logging.getLogger("crypto_tracker")

SOURCES = ("auto", "selenium", "api", "sample")
STATUS_OK = "ok"
STATUS_DEGRADED = "degraded"  # served by the API fallback
STATUS_REJECTED = "rejected"  # batch failed validation
STATUS_FAILED = "failed"  # no data at all


class PipelineError(RuntimeError):
    """Raised when no source could deliver a batch."""


def short_error(error: Any, limit: int = 300) -> str:
    """Collapse an exception into one tidy line for logs and the runs table.

    Selenium errors carry multi-line chromedriver stack traces that would
    otherwise swamp the SQLite ``runs.error`` column and the dashboard.
    """
    text = str(error).replace("\r", "\n")
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text.strip())
    return first_line if len(first_line) <= limit else first_line[: limit - 1] + "\u2026"


# ---------------------------------------------------------------------------
# stage 11 - logging
# ---------------------------------------------------------------------------
def setup_logging(cfg: Any, verbose: bool = False) -> logging.Logger:
    """Console + rotating file logging, configured entirely from config.yaml."""
    log_dir = cfg.path("logging.dir")
    log_dir.mkdir(parents=True, exist_ok=True)
    level = getattr(logging, str(cfg.get("logging.level", "INFO")).upper(), logging.INFO)
    if verbose:
        level = logging.DEBUG

    logger = logging.getLogger("crypto_tracker")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        log_dir / str(cfg.get("logging.file", "crypto_tracker.log")),
        maxBytes=int(cfg.get("logging.max_bytes", 1_048_576)),
        backupCount=int(cfg.get("logging.backup_count", 3)),
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


# ---------------------------------------------------------------------------
# stage 3 + 4 - acquisition
# ---------------------------------------------------------------------------
def acquire_records(cfg: Any, source: str, log: logging.Logger) -> tuple[list[dict[str, Any]], str, list[str]]:
    """Return ``(records, used_source, notes)``, falling back as configured."""
    top_n = int(cfg.get("scraper.top_n", 10))
    notes: list[str] = []

    if source in ("auto", "selenium"):
        try:
            log.info("scraping %s with Selenium", cfg.get("scraper.url"))
            records = CoinMarketCapScraper(cfg, logger=log).fetch_top()
            if records:
                return records, "selenium", notes
            notes.append("selenium returned zero rows")
        except Exception as exc:
            log.error("selenium failed: %s", short_error(exc))
            notes.append(f"selenium failed: {short_error(exc)}")
            if source == "selenium":
                raise PipelineError(f"selenium scraping failed: {short_error(exc)}") from exc

    if source in ("auto", "api"):
        try:
            log.info("using the API fallback")
            records, provider = FallbackFetcher(cfg, logger=log).fetch_top(top_n)
            return records, "api", notes + [f"provider={provider}"]
        except Exception as exc:
            log.error("fallback failed: %s", short_error(exc))
            notes.append(f"fallback failed: {short_error(exc)}")
            if source == "api":
                raise PipelineError(f"API fallback failed: {short_error(exc)}") from exc

    if source == "sample":
        log.info("using the offline sample source")
        # The seed moves with the clock so repeated demo runs build up a
        # realistic (non-flat) history instead of replaying one batch.
        return sample_records(top_n, seed=int(time.time()) % 100_000), "sample", notes

    raise PipelineError("no source produced data -> " + "; ".join(notes))


# ---------------------------------------------------------------------------
# the pipeline itself
# ---------------------------------------------------------------------------
def run_once(
    cfg: Any,
    source: str = "auto",
    log: logging.Logger | None = None,
    storage: Storage | None = None,
) -> dict[str, Any]:
    """Execute one full pass and return a summary dict (never raises for data issues)."""
    log = log or LOGGER
    started_at = utc_timestamp()
    started = time.perf_counter()
    storage = storage or build_storage(cfg, logger=log)
    top_n = int(cfg.get("scraper.top_n", 10))

    result: dict[str, Any] = {"started_at": started_at, "source": source, "status": STATUS_FAILED, "rows": 0}

    try:
        records, used_source, notes = acquire_records(cfg, source, log)
    except Exception as exc:
        log.error("acquisition failed: %s", short_error(exc))
        storage.record_run(STATUS_FAILED, rows=0, source=source, error=short_error(exc), started_at=started_at)
        result.update(
            status=STATUS_FAILED, error=short_error(exc), duration_s=round(time.perf_counter() - started, 2)
        )
        return result

    result["used_source"] = used_source
    result["notes"] = notes

    # Previous prices are read *before* saving so spike detection compares
    # against the genuinely previous run.
    previous_prices = storage.previous_prices([str(r.get("symbol")) for r in records if r.get("symbol")])

    enforce_exact = bool(cfg.get("validation.enforce_exact_rows", True))
    min_rows = int(cfg.get("validation.min_rows", 1))
    expected_rows = top_n if enforce_exact and used_source != "api" else None

    try:
        frame, report = clean_and_validate(
            records, expected_rows=expected_rows, timestamp=started_at, min_rows=min_rows
        )
    except BadBatchError as exc:
        log.error("batch rejected, nothing stored: %s", short_error(exc))
        storage.record_run(
            STATUS_REJECTED, rows=0, source=used_source, error=short_error(exc), started_at=started_at
        )
        result.update(
            status=STATUS_REJECTED,
            error=short_error(exc),
            report=getattr(exc, "report", None).summary() if getattr(exc, "report", None) else None,
            duration_s=round(time.perf_counter() - started, 2),
        )
        return result

    for warning in report.warnings:
        log.warning("data warning: %s", warning)

    stored = storage.save(frame)
    log.info(
        "stored %d rows (source=%s, new sqlite rows=%d) -> %s",
        stored["rows"],
        used_source,
        stored["inserted"],
        stored["csv_path"],
    )

    alerts: list[Any] = []
    if bool(cfg.get("alerts.enabled", True)):
        from alerts import AlertEngine

        alerts = AlertEngine(cfg, storage=storage, logger=log).run(frame, previous_prices=previous_prices)

    status = STATUS_DEGRADED if used_source == "api" else STATUS_OK
    duration = round(time.perf_counter() - started, 2)
    storage.record_run(
        status, rows=len(frame), source=used_source, error=None, started_at=started_at
    )
    log.info("run finished: status=%s rows=%d source=%s alerts=%d in %.2fs", status, len(frame), used_source, len(alerts), duration)

    result.update(
        status=status,
        rows=len(frame),
        inserted=stored["inserted"],
        csv_path=str(stored["csv_path"]),
        alerts=len(alerts),
        warnings=report.warnings,
        duration_s=duration,
        report=report.summary(),
    )
    return result


# ---------------------------------------------------------------------------
# stage 7 - scheduling
# ---------------------------------------------------------------------------
def run_schedule(cfg: Any, source: str, log: logging.Logger, interval_minutes: float | None = None) -> None:
    """Blocking scheduler loop; Ctrl+C shuts down cleanly."""
    import schedule

    interval = float(interval_minutes or cfg.get("scheduler.interval_minutes", 15))
    log.info("scheduler started: every %.1f minute(s), source=%s", interval, source)

    def job() -> None:
        summary = run_once(cfg, source=source, log=log)
        if summary["status"] in (STATUS_FAILED, STATUS_REJECTED):
            log.warning("scheduled run needs attention: %s", summary.get("error"))

    schedule.every(interval).minutes.do(job)
    if bool(cfg.get("scheduler.run_immediately", True)):
        job()
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:  # pragma: no cover - interactive
        log.info("scheduler stopped by user")


def print_status(cfg: Any, storage: Storage) -> None:
    health = storage.health_summary()
    print("=== scraper health ===")
    for key, value in health.items():
        print(f"{key:<16}: {value}")
    alerts = storage.recent_alerts(limit=10)
    print("\n=== last alerts ===")
    if alerts.empty:
        print("none yet")
    else:
        for row in alerts.itertuples(index=False):
            print(f"{row.created_at}  [{row.severity}] {row.kind:<9} {row.message}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cryptocurrency price tracker pipeline")
    parser.add_argument("--config", default=None, help="path to config.yaml")
    parser.add_argument("--source", choices=SOURCES, default="auto", help="data source (default: auto)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--once", action="store_true", help="run a single pass (default)")
    mode.add_argument("--schedule", action="store_true", help="run continuously on the configured interval")
    mode.add_argument("--status", action="store_true", help="print scraper health and exit")
    parser.add_argument("--interval", type=float, default=None, help="override scheduler.interval_minutes")
    parser.add_argument("--dry-run", action="store_true", help="force alerts into dry-run mode")
    parser.add_argument("--no-alerts", action="store_true", help="skip alert evaluation entirely")
    parser.add_argument("--verbose", action="store_true", help="debug-level console logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    from config import load_config

    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.dry_run:
        cfg = cfg.with_overrides({"alerts.dry_run": True})
    if args.no_alerts:
        cfg = cfg.with_overrides({"alerts.enabled": False})

    log = setup_logging(cfg, verbose=args.verbose)
    storage = build_storage(cfg, logger=log)

    if args.status:
        print_status(cfg, storage)
        return 0

    if args.schedule:
        run_schedule(cfg, args.source, log, interval_minutes=args.interval)
        return 0

    summary = run_once(cfg, source=args.source, log=log, storage=storage)
    print(
        f"status={summary['status']} rows={summary.get('rows', 0)} "
        f"source={summary.get('used_source', summary['source'])} "
        f"alerts={summary.get('alerts', 0)} duration={summary.get('duration_s', 0)}s"
    )
    if summary.get("error"):
        print(f"error: {summary['error']}")
    return 0 if summary["status"] in (STATUS_OK, STATUS_DEGRADED) else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
