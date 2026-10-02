"""Stage 10 - Streamlit dashboard.

Reads everything from SQLite (never scrapes in the render path) and refreshes
itself every ``dashboard.refresh_seconds``::

    streamlit run dashboard.py
"""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from analytics import correlation_matrix, pivot_prices
from config import load_config
from storage import build_storage

st.set_page_config(page_title="Crypto Price Tracker", page_icon="📈", layout="wide")

STATUS_COLORS = {"ok": "🟢", "degraded": "🟠", "rejected": "🟡", "failed": "🔴"}


@st.cache_resource
def get_storage():
    cfg = load_config()
    logging.getLogger("crypto_tracker").setLevel(logging.WARNING)
    return cfg, build_storage(cfg)


def render_data(storage, coin: str, hours: int, top_n: int) -> None:
    """Data panels; wrapped in a fragment so they auto-refresh."""
    snapshot = storage.latest_snapshot()

    st.subheader("Live top-10 snapshot")
    if snapshot.empty:
        st.info(
            "No data stored yet. Run `python main.py --once` (or use *Fetch now* in the sidebar), "
            "then come back."
        )
        return

    shown = snapshot.head(top_n).copy()
    shown = shown.rename(
        columns={"coin": "Coin", "symbol": "Symbol", "price": "Price ($)", "change_24h": "24h %", "market_cap": "Market cap ($)", "source": "Source"}
    )
    st.dataframe(
        shown,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Price ($)": st.column_config.NumberColumn(format="%.6g"),
            "24h %": st.column_config.NumberColumn(format="%+.2f%%"),
            "Market cap ($)": st.column_config.NumberColumn(format="%.4g"),
        },
    )

    left, right = st.columns([1, 1])
    with left:
        st.subheader(f"Top movers (24h)")
        movers = snapshot.dropna(subset=["change_24h"]).set_index("symbol")["change_24h"].astype(float)
        if not movers.empty:
            st.bar_chart(movers.sort_values(ascending=False))
        else:
            st.caption("no 24h change data in this batch")
    with right:
        st.subheader("Batch info")
        st.markdown(
            f"- batch timestamp: `{snapshot['timestamp'].iloc[0]}`\n"
            f"- data source: **{snapshot['source'].iloc[0]}**\n"
            f"- coins in batch: **{len(snapshot)}**"
        )

    st.subheader(f"Price history — {coin}")
    history = storage.history(symbol=coin, hours=hours)
    if history.empty:
        st.caption(f"no history for {coin} in the last {hours}h")
    else:
        wide = pivot_prices(history)
        if wide.empty:
            st.caption("not enough history for a chart")
        else:
            if wide.shape[1] > 1:
                wide = wide[[coin]] if coin in wide.columns else wide
            st.line_chart(wide)

    with st.expander("Correlation of returns (all coins, needs ≥ 2 points per coin)"):
        matrix = correlation_matrix(pivot_prices(storage.history(hours=hours)))
        if matrix.empty:
            st.caption("not enough history yet")
        else:
            st.dataframe(matrix.round(3), use_container_width=True)

    st.subheader("Alert log")
    alerts = storage.recent_alerts(limit=50)
    if alerts.empty:
        st.caption("no alerts recorded")
    else:
        st.dataframe(
            alerts[["created_at", "severity", "kind", "symbol", "message", "channel"]],
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Scraper health (last 20 runs)")
    runs = storage.recent_runs(limit=20)
    if not runs.empty:
        runs = runs.assign(status=runs["status"].map(lambda s: f"{STATUS_COLORS.get(s, '⚪')} {s}"))
    st.dataframe(runs, use_container_width=True, hide_index=True)


def render_fetch_now(cfg, storage) -> None:
    source = st.sidebar.selectbox("Manual fetch source", ("sample", "auto", "api"), index=0)
    if st.sidebar.button("Fetch now", use_container_width=True):
        from main import run_once

        with st.spinner("Running one pipeline pass…"):
            summary = run_once(cfg, source=source)
        if summary.get("status") in {"ok", "degraded"}:
            st.sidebar.success(f"stored {summary.get('rows', 0)} rows via {summary.get('used_source')}")
        else:
            st.sidebar.error(f"run {summary.get('status')}: {summary.get('error', 'unknown error')}")


def main() -> None:
    cfg, storage = get_storage()
    refresh = max(5, int(cfg.get("dashboard.refresh_seconds", 30)))
    default_coin = str(cfg.get("dashboard.default_coin", "BTC")).upper()
    history_hours = int(cfg.get("dashboard.history_hours", 168))
    top_n = int(cfg.get("scraper.top_n", 10))

    st.title("📈 Cryptocurrency Price Tracker")
    st.caption(f"auto-refresh every {refresh}s · data from SQLite · sources: selenium → API fallback")

    health = storage.health_summary()
    metrics = st.columns(5)
    last_status = health.get("last_status")
    metrics[0].metric("Rows stored", health.get("rows", 0))
    metrics[1].metric("Last run", f"{STATUS_COLORS.get(last_status, '⚪')} {last_status or 'n/a'}")
    metrics[2].metric("Last source", health.get("last_source") or "n/a")
    metrics[3].metric("Fallback runs (last 20)", health.get("fallback_runs", 0))
    metrics[4].metric("Failed runs (last 20)", health.get("failed_runs", 0))

    with st.sidebar:
        st.header("Controls")
        coin = st.text_input("Chart coin", value=default_coin).upper()
        hours = st.slider("History window (hours)", 1, 720, value=history_hours)
        st.divider()
        render_fetch_now(cfg, storage)

    fragment = st.fragment(run_every=f"{refresh}s")(render_data)
    fragment(storage, coin, hours, top_n)


if __name__ == "__main__":  # pragma: no cover
    main()
