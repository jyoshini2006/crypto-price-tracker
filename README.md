# 📈 Cryptocurrency Price Tracker

An end-to-end data pipeline that tracks the **top 10 cryptocurrencies**:
Selenium scrapes CoinMarketCap, an API takes over when the scraper fails, every
batch is validated before storage, alerts fire on thresholds/spikes/movers, and
a Streamlit dashboard reads the history from SQLite.

```text
scheduler → scraper (Selenium) → cleaner/validator → storage (CSV + SQLite) → alerts → dashboard
                     ↓ (on failure)
              API fallback (CoinGecko → CoinMarketCap) + logging
```

Every stage of the brief maps to one module — nothing is hardcoded in the
scraper; all settings live in [config.yaml](config.yaml).

| Stage | Module | What it does |
| ----- | ------ | ------------ |
| 1. Configuration | [config.py](config.py) | Loads `config.yaml`, supports `${VAR}` / `${VAR:-default}` env interpolation and `.env` secrets |
| 2. Driver setup | [scraper.py](scraper.py) | Headless Chrome (`--no-sandbox`, custom UA, images blocked), `webdriver_manager`, context-managed driver that always quits |
| 3. Resilient scraping | [scraper.py](scraper.py) | `WebDriverWait` on table rows (no `time.sleep` waits), one lazy-load scroll, `__NEXT_DATA__` JSON parsing before DOM selectors, 3 attempts with exponential backoff |
| 4. Fallback source | [fallback.py](fallback.py) | CoinGecko (key-less) → CoinMarketCap Pro (if `CMC_API_KEY` is set); every row tagged with a `source` column |
| 5. Cleaning & validation | [cleaner.py](cleaner.py) | Strips `$ , %`, expands `B/M/T`, casts to float, UTC timestamp; rejects bad batches (non-positive price, wrong row count, duplicates) with a `ValidationReport` |
| 6. Storage | [storage.py](storage.py) | Append-only CSV (header only when new, monthly rotation to `data/archive/`) **plus** SQLite (`prices`, `runs`, `alerts` tables) |
| 7. Scheduling | [main.py](main.py) | `schedule`-based loop (interval from config), or point cron / Windows Task Scheduler / GitHub Actions at `python main.py --once` |
| 8. Alerts | [alerts.py](alerts.py) | Price thresholds, top-3 gainers/losers, spike detection vs. the previous run; Telegram and email delivery, `dry_run` mode by default |
| 9. Analytics | [analytics.py](analytics.py) | SMA/EMA, rolling volatility (annualised), return correlations, and a clearly-labelled experimental next-interval forecast |
| 10. Dashboard | [dashboard.py](dashboard.py) | Streamlit: live top-10, price history, movers bar chart, alert log, scraper health; auto-refreshes from SQLite |
| 11. Logging & monitoring | [main.py](main.py) + [storage.py](storage.py) | Rotating file log + `runs` table (status, rows, source, error); fallback and failure runs are flagged |

---

## 🚀 Quick start

```bash
# from the repository root (Windows)
python -m venv venv
venv\Scripts\activate
pip install -r crypto_tracker/requirements.txt

cd crypto_tracker
python main.py --once                 # one pass: Selenium → API fallback
python main.py --source sample --once # offline demo, no Chrome needed
python main.py --schedule             # loop every scheduler.interval_minutes
python main.py --status               # scraper health + last alerts
python analytics.py --symbol BTC --forecast --correlation
streamlit run dashboard.py            # the dashboard
```

### Tests

```bash
cd crypto_tracker
python -m pytest            # 118 tests, no network or Chrome required
```

The suite covers parsing/validation, CSV rotation, SQLite semantics, all three
alert rules, fallback provider ordering, scraper payload/DOM parsing, retry
behaviour and the full pipeline (including rejection and degraded paths) with
fakes — it never touches the live site.

---

## ⚙️ Configuration

Everything is in [config.yaml](config.yaml): URL, top-N, headless flag, retries,
selectors, thresholds, interval, output paths. Secrets stay out of it — copy
[.env.example](.env.example) to `.env` and fill in only what you use:

| Variable | Purpose |
| -------- | ------- |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` | Telegram alert delivery |
| `SMTP_USERNAME`, `SMTP_PASSWORD`, `SMTP_FROM`, `SMTP_TO` | Email alerts |
| `CMC_API_KEY` | Optional CoinMarketCap Pro fallback |

Alerts default to `alerts.dry_run: true` (logged, not sent) — flip it after the
credentials are in.

## 📊 Data layout

```text
crypto_tracker/
├── data/
│   ├── prices.csv      # append-only log (the spec's core requirement)
│   ├── crypto.db       # SQLite: prices / runs / alerts
│   └── archive/        # prices_YYYY-MM.csv after monthly rotation
└── logs/
    ├── crypto_tracker.log   # rotating pipeline log
    ├── alerts.log           # one line per alert
    └── runs live in SQLite for the dashboard's health panel
```

CSV columns: `coin, symbol, price, change_24h, market_cap, timestamp, source`.

---

## 🖥️ Verification performed on this machine

- `118/118` tests pass (Python 3.14, pandas 3.0.6).
- Live API run: 10 real rows stored, alerts evaluated, run flagged `degraded`
  (status colours: 🟢 ok · 🟠 degraded · 🟡 rejected · 🔴 failed).
- Live Selenium run: Chrome/chromedriver launch and the retry/backoff chain work;
  `coinmarketcap.com` itself was **unreachable from this network**
  (`ERR_CONNECTION_TIMED_OUT`, confirmed with a plain HTTPS probe) while the API
  answered — exactly the scenario the fallback exists for.
- Dashboard verified running against SQLite with 8 batches of stored history.

> Scraping CoinMarketCap is inherently fragile: the DOM changes, bots get
> challenged, and some networks block the site outright. The `__NEXT_DATA__`
> parser, config-driven selectors and the API fallback are the mitigations —
> and every stored row's `source` column tells you which path produced it.

## 🤖 Optional ML extension

`analytics.forecast_next` offers `ema`, `linear` and `drift` estimators for the
next interval, with a ±1.96σ band. It is an experiment for the report, not a
prediction tool — the dashboard and CLI label it as such.
