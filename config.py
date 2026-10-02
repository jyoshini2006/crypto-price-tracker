"""Configuration layer for the crypto tracker.

All settings come from ``config.yaml``. Values may reference environment
variables using ``${VAR}`` or ``${VAR:-default}`` so that secrets (Telegram
token, SMTP password, CMC API key) never have to live in the YAML file.

Usage::

    from config import load_config
    cfg = load_config()
    top_n = cfg.get("scraper.top_n", 10)
    csv_path = cfg.path("storage.csv_path")
"""

from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any

import yaml

try:  # python-dotenv is optional: env vars work fine without it
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared in requirements.txt
    load_dotenv = None  # type: ignore[assignment]

PACKAGE_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = PACKAGE_ROOT / "config.yaml"

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")

REQUIRED_SECTIONS = ("scraper", "fallback", "storage", "alerts", "scheduler", "logging", "dashboard")


class ConfigError(RuntimeError):
    """Raised when the configuration file is missing or malformed."""


class Config:
    """Thin, read-only wrapper around the parsed YAML tree."""

    def __init__(self, data: dict[str, Any], path: Path | None = None) -> None:
        self._data = data
        # Named ``config_path`` (not ``path``) so it cannot shadow Config.path().
        self.config_path = path

    # -- reading ----------------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        """Return a nested value using dotted notation (``"scraper.top_n"``)."""
        node: Any = self._data
        for part in dotted_key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def require(self, dotted_key: str) -> Any:
        """Like :meth:`get` but raises when the key is absent."""
        sentinel = object()
        value = self.get(dotted_key, sentinel)
        if value is sentinel:
            raise ConfigError(f"missing required config key: {dotted_key}")
        return value

    def path(self, dotted_key: str, default: str | None = None) -> Path:
        """Return a filesystem path, resolved against the package directory."""
        raw = self.get(dotted_key, default)
        if raw in (None, ""):
            raise ConfigError(f"missing required path in config: {dotted_key}")
        return self.resolve_path(str(raw))

    def section(self, name: str) -> dict[str, Any]:
        value = self.get(name, {})
        return value if isinstance(value, dict) else {}

    def as_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def resolve_path(raw: str, base: Path | None = None) -> Path:
        """Absolute paths pass through; relative ones hang off the package root."""
        candidate = Path(os.path.expanduser(raw))
        if candidate.is_absolute():
            return candidate
        return (base or PACKAGE_ROOT) / candidate

    def with_overrides(self, overrides: dict[str, Any]) -> "Config":
        """Return a copy with dotted-key overrides applied (used by the CLI)."""
        data = copy.deepcopy(self._data)
        for dotted_key, value in overrides.items():
            if value is None:
                continue
            node = data
            parts = dotted_key.split(".")
            for part in parts[:-1]:
                node = node.setdefault(part, {})
            node[parts[-1]] = value
        return Config(data, self.config_path)


def _expand_env(value: Any, environ: dict[str, str]) -> Any:
    """Recursively substitute ``${VAR}`` / ``${VAR:-default}`` placeholders."""
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            return environ.get(name, default if default is not None else "")

        return _ENV_PATTERN.sub(replace, value)
    if isinstance(value, dict):
        return {key: _expand_env(item, environ) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env(item, environ) for item in value]
    return value


def load_config(
    path: str | Path | None = None,
    env_file: str | Path | None = None,
    environ: dict[str, str] | None = None,
) -> Config:
    """Load ``config.yaml``, apply ``.env`` and interpolate environment variables."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = PACKAGE_ROOT / config_path

    dotenv_path = Path(env_file) if env_file else PACKAGE_ROOT / ".env"
    if load_dotenv is not None and dotenv_path.exists():
        load_dotenv(dotenv_path=dotenv_path, override=False)

    if not config_path.exists():
        raise ConfigError(f"config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:  # pragma: no cover - malformed user file
        raise ConfigError(f"could not parse {config_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping at the top level")

    data = _expand_env(raw, environ if environ is not None else dict(os.environ))
    missing = [section for section in REQUIRED_SECTIONS if section not in data]
    if missing:
        raise ConfigError(f"{config_path} is missing required sections: {', '.join(missing)}")

    return Config(data, config_path)
