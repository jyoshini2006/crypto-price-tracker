"""Shared fixtures: a config redirected to a temp dir, and sample batches."""

import pytest

from cleaner import records_to_dataframe
from config import load_config
from fallback import sample_records


@pytest.fixture
def cfg(tmp_path):
    """The real config.yaml, with storage/logging/alerts pointed at tmp_path."""
    base = load_config()
    return base.with_overrides(
        {
            "storage.csv_path": str(tmp_path / "data" / "prices.csv"),
            "storage.sqlite_path": str(tmp_path / "data" / "crypto.db"),
            "storage.archive_dir": str(tmp_path / "data" / "archive"),
            "logging.dir": str(tmp_path / "logs"),
            "alerts.dry_run": True,
            "alerts.enabled": True,
        }
    )


@pytest.fixture
def sample_batch():
    """A validated-looking frame of ten sample coins."""
    return records_to_dataframe(sample_records(10, seed=3))
