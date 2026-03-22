"""
run_backtest_multi.py
=====================
Download historical OHLCV data for multiple assets across all major market
classes (crypto, forex, stocks, commodities, indices) and run the full ICT
backtest on each asset.

Data sources (all free, no API key required)
--------------------------------------------
* Crypto   → ccxt / Binance  (BTC/USDT, ETH/USDT, …)
* Forex    → yfinance         (EURUSD=X, GBPUSD=X, …)
* Stocks   → yfinance         (AAPL, NVDA, TSLA, …)
* Commodities → yfinance      (GC=F gold, CL=F crude, SI=F silver, …)
* Indices  → yfinance         (^GSPC S&P500, ^NDX Nasdaq, ^DJI Dow, …)

Usage
-----
    python run_backtest_multi.py                          # default mixed basket
    python run_backtest_multi.py --assets BTC/USDT AAPL GC=F ^GSPC EURUSD=X
    python run_backtest_multi.py --exchange bybit --bars 2000 --no-plot
    python run_backtest_multi.py --class crypto           # only crypto assets
    python run_backtest_multi.py --class stock index      # stocks + indices

Note: yfinance 5m data is capped at 60 days; 1h data at ~730 days.
      4h bars are synthesised by resampling 1h OHLCV for yfinance assets.
"""

from __future__ import annotations

import argparse
import io
import sys
import warnings
from pathlib import Path

# Force UTF-8 output on Windows (avoids charmap errors with → ← etc.)
if sys.stdout and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

# ── Data directory ────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent / "backtest_data"


# ══════════════════════════════════════════════════════════════════════════════
#  Asset registry
#  Each entry: display_name → {source, yf_ticker, instrument, asset_class}
#  source    : "ccxt" | "yfinance"
#  instrument: cost model used in backtest.py
#  asset_class: used for --class filter
# ══════════════════════════════════════════════════════════════════════════════

ASSET_REGISTRY: dict[str, dict] = {
    # ── Crypto (ccxt / Binance — free, deep history) ──────────────────────
    "BTC/USDT":  {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    "ETH/USDT":  {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    "SOL/USDT":  {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    "XRP/USDT":  {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    "BNB/USDT":  {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    "DOGE/USDT": {"source": "ccxt",     "instrument": "crypto",    "class": "crypto"},
    # ── Forex (yfinance — free) ────────────────────────────────────────────
    "EURUSD=X":  {"source": "yfinance", "instrument": "forex",     "class": "forex"},
    "GBPUSD=X":  {"source": "yfinance", "instrument": "forex",     "class": "forex"},
    "USDJPY=X":  {"source": "yfinance", "instrument": "forex",     "class": "forex"},
    "AUDUSD=X":  {"source": "yfinance", "instrument": "forex",     "class": "forex"},
    "USDCAD=X":  {"source": "yfinance", "instrument": "forex",     "class": "forex"},
    # ── Stocks (yfinance) ─────────────────────────────────────────────────
    "AAPL":      {"source": "yfinance", "instrument": "stock",     "class": "stock"},
    "NVDA":      {"source": "yfinance", "instrument": "stock",     "class": "stock"},
    "TSLA":      {"source": "yfinance", "instrument": "stock",     "class": "stock"},
    "MSFT":      {"source": "yfinance", "instrument": "stock",     "class": "stock"},
    "AMZN":      {"source": "yfinance", "instrument": "stock",     "class": "stock"},
    # ── Commodities (yfinance — all are exchange-traded futures, tight spreads) ─
    "GC=F":      {"source": "yfinance", "instrument": "futures",   "class": "commodity"},
    "CL=F":      {"source": "yfinance", "instrument": "futures",   "class": "commodity"},
    "SI=F":      {"source": "yfinance", "instrument": "futures",   "class": "commodity"},
    "NG=F":      {"source": "yfinance", "instrument": "futures",   "class": "commodity"},
    # ── Indices (yfinance) ────────────────────────────────────────────────
    "^GSPC":     {"source": "yfinance", "instrument": "index",       "class": "index"},
    "^NDX":      {"source": "yfinance", "instrument": "index",       "class": "index"},
    "^DJI":      {"source": "yfinance", "instrument": "index",       "class": "index"},
    "^FTSE":     {"source": "yfinance", "instrument": "index",       "class": "index"},
    # ^N225 excluded: Nikkei trades 00:00-06:30 UTC, zero overlap with kill zones
    # ── Financial futures (yfinance — E-mini equity + bond futures) ────────
    # Use fin_futures cost model: ~0.005%/side (vs 0.03% for commodities)
    # because commission is ~$2-5/contract on a large notional, not % of price
    "ES=F":      {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # E-mini S&P 500
    "NQ=F":      {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # E-mini Nasdaq 100
    "YM=F":      {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # E-mini Dow Jones
    "RTY=F":     {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # E-mini Russell 2000
    "ZB=F":      {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # 30-Year T-Bond
    "ZN=F":      {"source": "yfinance", "instrument": "fin_futures", "class": "futures"},  # 10-Year T-Note
    # ── Agricultural futures (yfinance) ────────────────────────────────────
    "ZC=F":      {"source": "yfinance", "instrument": "futures",     "class": "futures"},  # Corn
    "ZW=F":      {"source": "yfinance", "instrument": "futures",     "class": "futures"},  # Wheat
    "ZS=F":      {"source": "yfinance", "instrument": "futures",     "class": "futures"},  # Soybeans
}

# Default mixed basket — one from each asset class
DEFAULT_ASSETS = [
    "BTC/USDT",   # crypto
    "ETH/USDT",   # crypto
    "EURUSD=X",   # forex
    "GBPUSD=X",   # forex
    "AAPL",       # stock
    "NVDA",       # stock
    "GC=F",       # commodity (gold)
    "CL=F",       # commodity (crude oil)
    "^GSPC",      # index (S&P 500)
    "^NDX",       # index (Nasdaq 100)
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _symbol_to_prefix(symbol: str) -> str:
    """Safe filename prefix from a symbol string."""
    return symbol.replace("/", "_").replace("^", "").replace("=", "_").replace(" ", "_")


def _get_meta(symbol: str, override_instrument: str | None = None) -> dict:
    """Return registry entry for symbol, falling back to sensible defaults."""
    meta = ASSET_REGISTRY.get(symbol, {
        "source": "ccxt",
        "instrument": "crypto",
        "class": "crypto",
    }).copy()
    if override_instrument:
        meta["instrument"] = override_instrument
    return meta


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1a: download via ccxt (crypto)
# ══════════════════════════════════════════════════════════════════════════════

# Milliseconds per bar for each timeframe (used for pagination offset)
_TF_MS: dict[str, int] = {"4h": 4 * 3600 * 1000, "1h": 3600 * 1000, "5m": 5 * 60 * 1000}


def _ccxt_fetch_paginated(exchange, symbol: str, tf_key: str, total_bars: int) -> list:
    """
    Fetch up to `total_bars` OHLCV rows by paginating backwards from the present.
    Binance and most ccxt exchanges cap a single fetch_ohlcv call at 1000 rows.
    """
    per_page = 1000
    ms_per_bar = _TF_MS.get(tf_key, 5 * 60 * 1000)
    pages: list[list] = []
    since: int | None = None

    while sum(len(p) for p in pages) < total_bars:
        chunk = exchange.fetch_ohlcv(
            symbol, timeframe=tf_key, limit=per_page,
            **({"since": since} if since is not None else {}),
        )
        if not chunk:
            break
        pages.insert(0, chunk)
        # Move the window further back in time
        since = chunk[0][0] - ms_per_bar * per_page
        if len(chunk) < per_page:
            break  # reached the beginning of available history

    # Flatten → deduplicate by timestamp → sort ascending → keep most recent N
    seen: dict[int, list] = {}
    for page in pages:
        for row in page:
            seen[row[0]] = row
    return sorted(seen.values(), key=lambda x: x[0])[-total_bars:]


def download_asset_ccxt(
    symbol: str, exchange, bars: int, bars_5m: int, force: bool = False
) -> bool:
    """Download 4h/1h/5m via ccxt and cache to DATA_DIR CSVs.

    Parameters
    ----------
    bars     : bar count for 4h and 1h timeframes (≤ 1000 for a single request)
    bars_5m  : bar count for the 5m timeframe; paginated automatically when > 1000
    """
    prefix = _symbol_to_prefix(symbol)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tf_bars = {"4h": min(bars, 1000), "1h": min(bars, 1000), "5m": bars_5m}

    for tf_key in ("4h", "1h", "5m"):
        path = DATA_DIR / f"{prefix}_{tf_key}_data_clean.csv"
        if path.exists() and not force:
            print(f"  [{symbol}] {tf_key}: cached ({path.name})")
            continue

        n_bars = tf_bars[tf_key]
        print(f"  [{symbol}] {tf_key}: downloading {n_bars} bars …", end="", flush=True)
        try:
            if n_bars > 1000:
                ohlcv = _ccxt_fetch_paginated(exchange, symbol, tf_key, n_bars)
            else:
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe=tf_key, limit=n_bars)
            if not ohlcv:
                print(" NO DATA")
                return False

            df = pd.DataFrame(
                ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
            )
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
            df.set_index("timestamp", inplace=True)
            df.sort_index(inplace=True)
            df.to_csv(path)
            print(f" {len(df)} bars → {path.name}")
        except Exception as e:
            print(f" ERROR: {e}")
            return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1b: download via yfinance (forex, stocks, commodities, indices)
# ══════════════════════════════════════════════════════════════════════════════

def download_asset_yfinance(symbol: str, force: bool = False) -> bool:
    """
    Download 1h and 5m data via yfinance, synthesise 4h by resampling 1h.
    Caches results as CSVs in DATA_DIR.

    yfinance limits:
      5m  → max 60 days
      1h  → max 730 days
    """
    try:
        import yfinance as yf
    except ImportError:
        print("ERROR: yfinance is not installed. Run: pip install yfinance")
        return False

    prefix = _symbol_to_prefix(symbol)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # ── Download 1h (used directly and resampled to 4h) ──────────────────
    path_1h = DATA_DIR / f"{prefix}_1h_data_clean.csv"
    path_4h = DATA_DIR / f"{prefix}_4h_data_clean.csv"

    if path_1h.exists() and path_4h.exists() and not force:
        print(f"  [{symbol}] 1h/4h: cached")
    else:
        print(f"  [{symbol}] 1h: downloading (2y) …", end="", flush=True)
        try:
            df1h = yf.download(
                symbol, period="2y", interval="1h",
                auto_adjust=True, progress=False, threads=False,
            )
            if df1h.empty:
                print(" NO DATA")
                return False

            df1h = _normalise_yf(df1h)
            df1h.to_csv(path_1h)
            print(f" {len(df1h)} bars → {path_1h.name}")

            # Resample 1h → 4h
            df4h = df1h.resample("4h").agg(
                {"open": "first", "high": "max", "low": "min",
                 "close": "last", "volume": "sum"}
            ).dropna(subset=["open"])
            df4h.to_csv(path_4h)
            print(f"  [{symbol}] 4h: resampled from 1h → {len(df4h)} bars → {path_4h.name}")
        except Exception as e:
            print(f" ERROR: {e}")
            return False

    # ── Download 5m (capped at 60d by yfinance) ───────────────────────────
    path_5m = DATA_DIR / f"{prefix}_5m_data_clean.csv"

    if path_5m.exists() and not force:
        print(f"  [{symbol}] 5m: cached ({path_5m.name})")
    else:
        print(f"  [{symbol}] 5m: downloading (60d) …", end="", flush=True)
        try:
            df5m = yf.download(
                symbol, period="60d", interval="5m",
                auto_adjust=True, progress=False, threads=False,
            )
            if df5m.empty:
                print(" NO DATA")
                return False

            df5m = _normalise_yf(df5m)
            df5m.to_csv(path_5m)
            print(f" {len(df5m)} bars → {path_5m.name}")
        except Exception as e:
            print(f" ERROR: {e}")
            return False

    return True


def _normalise_yf(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a yfinance DataFrame to the pipeline's expected format."""
    df = df.copy()

    # yfinance sometimes returns MultiIndex columns (ticker, field)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.columns = [c.strip().lower() for c in df.columns]

    # Rename yfinance column names to pipeline names
    rename = {"adj close": "close", "adj_close": "close"}
    df.rename(columns=rename, inplace=True)

    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"yfinance DataFrame missing columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Ensure UTC-aware DatetimeIndex named 'timestamp'
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    elif df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df.index.name = "timestamp"
    df.sort_index(inplace=True)
    df.dropna(inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Dispatcher: choose download method based on asset registry
# ══════════════════════════════════════════════════════════════════════════════

def download_asset(
    symbol: str, exchange, bars: int, bars_5m: int, force: bool = False
) -> bool:
    meta = _get_meta(symbol)
    if meta["source"] == "yfinance":
        return download_asset_yfinance(symbol, force=force)
    return download_asset_ccxt(symbol, exchange, bars, bars_5m, force=force)


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2: enrich + generate signals + backtest for one asset
# ══════════════════════════════════════════════════════════════════════════════

def _diagnose_conditions(enriched: dict) -> None:
    """Print per-condition hit rates on the 5m frame to explain zero signals."""
    from signal_engine import (
        _align_htf, _build_conditions, KILL_ZONES, CONDITION_NAMES,
        MIN_RR, MIN_RISK_ATR, _build_sl, _build_tp2,
    )
    import numpy as np

    df_5m = enriched.get("5m")
    df_4h = enriched.get("4h")
    if df_5m is None or df_4h is None:
        return

    htf_structure = _align_htf(df_4h, df_5m.index, ["structure"])["structure"]
    df_1h = enriched.get("1h")
    conds = _build_conditions(df_5m, df_4h, htf_structure, choch_lookback=10, df_1h=df_1h)

    n = len(df_5m)
    print(f"  ── Condition diagnostics (n={n} 5m bars) ──")
    print(f"  {'Condition':<18} {'Long%':>7} {'Short%':>8}")

    for name in CONDITION_NAMES:
        cl, cs = conds[name]
        print(f"  {name:<18} {cl.mean()*100:>6.1f}%  {cs.mean()*100:>6.1f}%")

    # Show session distribution
    if "session" in df_5m.columns:
        sess_counts = df_5m["session"].value_counts()
        kz_total = df_5m["session"].isin(KILL_ZONES).sum()
        print(f"  Kill-zone bars: {kz_total}/{n} ({kz_total/n*100:.1f}%)")
        for sess, cnt in sess_counts.items():
            tag = " ← active" if sess in KILL_ZONES else ""
            print(f"    {sess:<12} {cnt:>6} bars{tag}")

    # Long and short score distributions
    long_scores  = sum(conds[name][0].astype(np.int8) for name in CONDITION_NAMES)
    short_scores = sum(conds[name][1].astype(np.int8) for name in CONDITION_NAMES)
    for label, scores in [("Long", long_scores), ("Short", short_scores)]:
        for k in range(5, 8):
            cnt = int((scores >= k).sum())
            print(f"  {label} ≥{k}/7 conditions: {cnt} bars ({cnt/n*100:.2f}%)")

    # Show which condition is the "missing" one at 6/7 bars
    for label, scores, cond_idx in [("Long", long_scores, 0), ("Short", short_scores, 1)]:
        mask_6of7 = scores == 6
        if mask_6of7.sum() == 0:
            continue
        print(f"  {label} 6/7 bars — missing condition breakdown:")
        for name in CONDITION_NAMES:
            c = conds[name][cond_idx]
            missing = int((mask_6of7 & ~c).sum())
            if missing:
                print(f"    {name:<18} missing on {missing} bars")


def _forward_fill_htf_onto_exec(df_htf: pd.DataFrame, df_exec: pd.DataFrame) -> pd.DataFrame:
    """
    Forward-fill HTF zone/level columns onto the execution frame, then re-run
    sweep detection and PD arrays so all conditions are fresh on the exec frame.
    """
    from liquidity import detect_sweeps, detect_judas_swing
    from pd_arrays import add_pd_arrays

    htf_cols = [
        "range_high", "range_low", "equilibrium",
        "premium", "discount",
        "ote_low", "ote_high", "ote_short_low", "ote_short_high",
        "bsl_level", "ssl_level", "eqh_level", "eql_level",
    ]
    sub = df_htf[[c for c in htf_cols if c in df_htf.columns]].copy()

    for c in [c for c in ("premium", "discount") if c in sub.columns]:
        sub[c] = sub[c].fillna(False).astype(bool)
    for c in [c for c in htf_cols if c not in ("premium", "discount") and c in sub.columns]:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")

    with pd.option_context("future.no_silent_downcasting", True):
        aligned = sub.reindex(sub.index.union(df_exec.index)).ffill().reindex(df_exec.index)

    overlap = [c for c in aligned.columns if c in df_exec.columns]
    exec_clean = df_exec.drop(columns=overlap)
    df_exec = pd.concat([exec_clean, aligned], axis=1)

    df_exec = detect_sweeps(df_exec)
    df_exec = detect_judas_swing(df_exec)
    # Use the exec frame's own TF label so PD-array lookbacks are appropriate
    tf_label = "5m"  # signal_engine always looks for "5m" key; label doesn't change logic
    df_exec, _, _ = add_pd_arrays(df_exec, timeframe=tf_label)
    return df_exec


def run_asset(symbol: str, instrument: str, initial_equity: float, plot: bool,
              exec_tf: str = "auto",
              override_source: str | None = None,
              override_local_dir: str | None = None,
              override_local_prefix: str | None = None) -> dict:
    """
    Full pipeline for one asset.

    exec_tf : "5m" | "1h" | "auto"
        "5m"  — use 5m bars as execution frame (best for crypto with deep history)
        "1h"  — use 1h bars as execution frame (best for yfinance assets capped at 60d 5m)
        "auto" — "5m" for ccxt sources, "1h" for yfinance sources (default)
    override_source      : if set, overrides the registry source (e.g. "local")
    override_local_dir   : local CSV directory (used when override_source="local")
    override_local_prefix: local CSV filename prefix (used when override_source="local")
    """
    from data_feed import fetch_mtf
    from market_structure import add_market_structure
    from dealing_range import compute_dealing_range
    from liquidity import add_liquidity
    from pd_arrays import add_pd_arrays
    from signal_engine import generate_signals
    from backtest import full_wf_report

    prefix = override_local_prefix or _symbol_to_prefix(symbol)
    meta = _get_meta(symbol)
    asset_class = meta.get("class", "unknown")
    source = override_source or meta.get("source", "ccxt")

    # Resolve execution timeframe
    if exec_tf == "auto":
        effective_exec_tf = "5m" if source == "ccxt" else "1h"
    else:
        effective_exec_tf = exec_tf

    print(f"\n{'─'*60}")
    print(f"  Asset: {symbol}  ({asset_class})  instrument={instrument}  exec={effective_exec_tf}")
    print(f"{'─'*60}")

    # 1. Load MTF data from local CSVs
    local_dir = override_local_dir or str(DATA_DIR)
    try:
        mtf = fetch_mtf(
            symbol,
            source="local",
            local_dir=local_dir,
            local_prefix=prefix,
        )
    except FileNotFoundError as e:
        print(f"  SKIP — {e}")
        return {"error": str(e)}

    # 2. Enrich all available timeframes
    # pivot_n controls how many bars each side define a swing high/low.
    # Smaller values = more structure points = more CHoCH events.
    # 5m exec: pivot_n=5 (25 min each side)  1h exec: pivot_n=3 (3h each side, more sensitive)
    pivot_n_exec = 3 if effective_exec_tf == "1h" else 5
    enriched: dict[str, pd.DataFrame] = {}
    for tf, df in mtf.items():
        if effective_exec_tf == "1h" and tf == "5m":
            continue  # skip 5m enrichment when not needed
        pn = pivot_n_exec if tf in ("5m", "1h") else 5  # use smaller pivot only on exec frame
        df = add_market_structure(df, pivot_n=pn)
        df = compute_dealing_range(df, window="session")
        df = add_liquidity(df, lookback=20)
        df, _, _ = add_pd_arrays(df, timeframe=tf)
        enriched[tf] = df

    # 3. Forward-fill HTF zone/level columns onto execution frame + sweep re-detection
    if effective_exec_tf == "1h":
        # 1h exec: 4h provides the HTF reference levels for 1h execution bars
        enriched["1h"] = _forward_fill_htf_onto_exec(enriched["4h"], enriched["1h"])
        # Signal engine always looks for "5m" key; alias 1h as "5m"
        enriched["5m"] = enriched["1h"]
    else:
        # 5m exec (default): 1h provides the HTF reference levels for 5m execution bars
        enriched["5m"] = _forward_fill_htf_onto_exec(enriched["1h"], enriched["5m"])

    # 4. Generate signals
    # For 1h exec: relax min_risk_atr (stop distance filter) because 1h ATR is
    # much larger than 5m ATR — tight stops are natural, not a degenerate case.
    # Use signal_engine globals (which may have been overridden by --config).
    import signal_engine as _se_module
    sig_min_risk_atr = _se_module.MIN_RISK_ATR if effective_exec_tf != "1h" else min(
        _se_module.MIN_RISK_ATR, 0.5
    )
    signals = generate_signals(
        enriched,
        min_risk_atr=sig_min_risk_atr,
        min_rr=_se_module.MIN_RR,
        choch_lookback=_se_module.CHOCH_LOOKBACK,
        min_conditions=_se_module.MIN_CONDITIONS,
    )
    print(f"  Signals generated: {len(signals)}  "
          f"(long={int((signals['signal']==1).sum()) if not signals.empty else 0}, "
          f"short={int((signals['signal']==-1).sum()) if not signals.empty else 0})")

    if signals.empty:
        _diagnose_conditions(enriched)
        print("  No signals — skipping backtest.")
        return {"error": "no signals"}

    # 5. Backtest
    result = full_wf_report(
        enriched["5m"],
        signals,
        instrument=instrument,
        initial_equity=initial_equity,
        plot=plot,
    )
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Summary table
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(results: dict[str, dict]) -> None:
    print("\n" + "=" * 90)
    print("  MULTI-ASSET BACKTEST SUMMARY")
    print("=" * 90)

    header = (
        f"{'Asset':<14} {'Class':<11} {'Instr':<11}"
        f" {'Trades':>7} {'WinRate':>8} {'PF':>6} {'Sharpe':>7} {'MaxDD%':>8} {'Return%':>9}"
    )
    print(header)
    print("-" * 90)

    for symbol, res in results.items():
        stats = res.get("full", {})
        meta = _get_meta(symbol)
        asset_class = meta.get("class", "?")
        instrument  = meta.get("instrument", "?")

        if "error" in res or "error" in stats:
            err = res.get("error") or stats.get("error", "unknown")
            print(f"  {symbol:<12}  {asset_class:<9}  {instrument:<9}  ERROR: {err}")
            continue

        print(
            f"  {symbol:<12}  {asset_class:<9}  {instrument:<9}"
            f"  {stats.get('total_trades', 0):>7}"
            f"  {stats.get('win_rate', 0)*100:>7.1f}%"
            f"  {stats.get('profit_factor', 0):>6.2f}"
            f"  {stats.get('sharpe', 0):>7.2f}"
            f"  {stats.get('max_drawdown_pct', 0):>7.2f}%"
            f"  {stats.get('total_return_pct', 0):>8.2f}%"
        )

    print("=" * 90)

    # Walk-forward overfit flags
    overfit = [s for s, r in results.items() if r.get("walk_forward", {}).get("overfit_flag")]
    if overfit:
        print(f"\n  *** Overfit warning on: {', '.join(overfit)} ***")
    print()


# ══════════════════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Multi-asset ICT backtest runner (crypto/forex/stock/commodity/index)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python run_backtest_multi.py
  python run_backtest_multi.py --assets BTC/USDT AAPL GC=F EURUSD=X ^GSPC
  python run_backtest_multi.py --class crypto stock
  python run_backtest_multi.py --exchange bybit --bars 2000 --no-plot
  python run_backtest_multi.py --force-download

Registered symbols by class
----------------------------
  crypto   : BTC/USDT ETH/USDT SOL/USDT XRP/USDT BNB/USDT DOGE/USDT
  forex    : EURUSD=X GBPUSD=X USDJPY=X AUDUSD=X USDCAD=X
  stock    : AAPL NVDA TSLA MSFT AMZN
  commodity: GC=F CL=F SI=F NG=F
  index    : ^GSPC ^NDX ^DJI ^FTSE ^N225
  futures  : ES=F NQ=F YM=F RTY=F ZB=F ZN=F ZC=F ZW=F ZS=F
""",
    )
    ap.add_argument(
        "--assets", nargs="+", default=None,
        help="Explicit list of symbols to test (overrides --class)",
    )
    ap.add_argument(
        "--class", dest="asset_classes", nargs="+",
        choices=["crypto", "forex", "stock", "commodity", "index", "futures"],
        metavar="CLASS",
        help="Filter by asset class(es); ignored when --assets is set",
    )
    ap.add_argument("--exchange", default="binance",
                    help="ccxt exchange ID for crypto assets (default: binance)")
    ap.add_argument("--bars", type=int, default=1000,
                    help="Bars to fetch for 4h/1h ccxt timeframes (default: 1000)")
    ap.add_argument("--bars-5m", dest="bars_5m", type=int, default=17280,
                    help="5m bars to fetch for crypto (default: 17280 ≈ 60 days; paginated)")
    ap.add_argument("--equity", type=float, default=10_000.0,
                    help="Starting equity for backtest (default: 10000)")
    ap.add_argument(
        "--instrument", default=None,
        choices=["crypto", "forex", "stock", "commodity", "index", "futures", "fin_futures"],
        help="Override cost model for all assets (default: per-asset from registry)",
    )
    ap.add_argument("--no-plot", action="store_true",
                    help="Disable matplotlib charts")
    ap.add_argument("--force-download", action="store_true",
                    help="Re-download data even if cached CSVs exist")
    ap.add_argument(
        "--exec-tf", dest="exec_tf", default="auto",
        choices=["5m", "1h", "auto"],
        help="Execution timeframe: '5m' (crypto default), '1h' (yfinance default), 'auto'",
    )
    ap.add_argument(
        "--symbol", default=None,
        help="Single symbol to backtest using local data (e.g. NQ/USD). Overrides --assets.",
    )
    ap.add_argument(
        "--source", default=None,
        help="Override data source for --symbol (e.g. 'local')",
    )
    ap.add_argument(
        "--local-dir", dest="local_dir", default=None,
        help="Local CSV directory when --source local is used (e.g. nasdaq)",
    )
    ap.add_argument(
        "--local-prefix", dest="local_prefix", default=None,
        help="Local CSV filename prefix when --source local is used (e.g. nasdaq)",
    )
    ap.add_argument(
        "--config", default=None,
        help="Path to config.yaml — loads signal/risk parameters from file",
    )
    args = ap.parse_args()

    # ── Apply config.yaml parameters to signal_engine globals (if provided) ──
    if args.config:
        try:
            import yaml as _yaml
            with open(args.config, "r", encoding="utf-8") as _f:
                _cfg = _yaml.safe_load(_f) or {}
            import signal_engine as _se
            sig_cfg = _cfg.get("signal", {}) or {}
            _map_se = {
                "min_conditions": "MIN_CONDITIONS",
                "min_rr":         "MIN_RR",
                "sl_atr_buffer":  "SL_ATR_BUFFER",
                "default_rr_mult":"DEFAULT_RR_MULT",
                "min_risk_atr":   "MIN_RISK_ATR",
                "ema_period":     "EMA_PERIOD",
                "tp1_rr":         "TP1_RR",
                "choch_lookback": "CHOCH_LOOKBACK",
                "sweep_lookback": "SWEEP_LOOKBACK",
            }
            for yaml_key, module_attr in _map_se.items():
                if yaml_key in sig_cfg:
                    current = getattr(_se, module_attr)
                    setattr(_se, module_attr, type(current)(sig_cfg[yaml_key]))
                    print(f"  [config] {module_attr} = {getattr(_se, module_attr)}")
        except Exception as _e:
            print(f"  [config] WARNING: could not load {args.config}: {_e}")

    # ── Resolve asset list ────────────────────────────────────────────────────
    if args.symbol:
        target_symbols = [args.symbol]
    elif args.assets:
        target_symbols = args.assets
    elif args.asset_classes:
        target_symbols = [
            sym for sym, meta in ASSET_REGISTRY.items()
            if meta["class"] in args.asset_classes
        ]
    else:
        target_symbols = DEFAULT_ASSETS

    if not target_symbols:
        print("No assets selected. Use --assets or --class to specify assets.")
        sys.exit(1)

    # ── If using --symbol with --source local, skip download entirely ────────
    use_local_override = bool(args.symbol and args.source == "local")

    # ── Connect to ccxt exchange (needed for crypto only) ─────────────────────
    exchange = None
    has_crypto = (not use_local_override) and any(
        _get_meta(s)["source"] == "ccxt" for s in target_symbols
    )
    if has_crypto:
        try:
            import ccxt as _ccxt
        except ImportError:
            print("ERROR: ccxt is not installed. Run: pip install ccxt")
            sys.exit(1)

        ex_cls = getattr(_ccxt, args.exchange, None)
        if ex_cls is None:
            print(f"ERROR: Unknown exchange '{args.exchange}'. Check ccxt docs.")
            sys.exit(1)

        exchange = ex_cls({"enableRateLimit": True})
        print(f"\nConnected to {args.exchange.upper()} (crypto)")

    # Check yfinance is available for non-crypto assets
    has_yfinance_assets = (not use_local_override) and any(
        _get_meta(s)["source"] == "yfinance" for s in target_symbols
    )
    if has_yfinance_assets:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            print("ERROR: yfinance is not installed. Run: pip install yfinance")
            sys.exit(1)

    print(f"\nAssets to test ({len(target_symbols)}): {target_symbols}\n")

    # ── Download data (skip when using local override) ────────────────────────
    if use_local_override:
        print("  [local override] Skipping download — using local CSV data.")
        available = list(target_symbols)
    else:
        print("=" * 60)
        print("  DOWNLOADING HISTORICAL DATA")
        print("=" * 60)

        available = []
        for sym in target_symbols:
            ok = download_asset(sym, exchange, args.bars, args.bars_5m, force=args.force_download)
            if ok:
                available.append(sym)

        if not available:
            print("\nNo data downloaded. Exiting.")
            sys.exit(1)

    print(f"\nReady to backtest: {available}")

    # ── Run backtests ─────────────────────────────────────────────────────────
    results: dict[str, dict] = {}
    for sym in available:
        instrument = args.instrument or _get_meta(sym).get("instrument", "futures")
        try:
            res = run_asset(
                sym, instrument, args.equity,
                plot=not args.no_plot,
                exec_tf=args.exec_tf,
                override_source=args.source if use_local_override else None,
                override_local_dir=args.local_dir if use_local_override else None,
                override_local_prefix=args.local_prefix if use_local_override else None,
            )
            results[sym] = res
        except Exception as e:
            print(f"  [{sym}] FATAL: {e}")
            results[sym] = {"error": str(e)}

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
