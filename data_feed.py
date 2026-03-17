"""
data_feed.py
============
Multi-timeframe (MTF) data feed for an ICT Forex/Crypto trading bot.

Supported sources
-----------------
* ccxt.binance   – crypto symbols  (e.g. "BTC/USDT")
* oandapyV20     – Forex symbols   (e.g. "EUR_USD")
* local CSV      – offline / back-test mode

Entry point
-----------
    from data_feed import fetch_mtf

    # --- crypto (live) ---
    import ccxt
    ex = ccxt.binance()
    data = fetch_mtf("BTC/USDT", ex)

    # --- forex (live) ---
    from oandapyV20 import API
    ex = API(access_token="<token>", environment="practice")
    data = fetch_mtf("EUR_USD", ex)

    # --- local CSV back-test ---
    data = fetch_mtf("XAU/USD", source="local",
                     local_dir="gold_clean_data", local_prefix="XAU")

    data["4h"]   # pandas DataFrame with HTF bias
    data["1h"]   # intermediate
    data["5m"]   # LTF execution
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# ── optional heavy imports (graceful degradation) ─────────────────────────────
try:
    import ccxt
    _HAS_CCXT = True
except ImportError:
    _HAS_CCXT = False
    warnings.warn("ccxt not installed – crypto live feed unavailable.", stacklevel=1)

try:
    import oandapyV20
    import oandapyV20.endpoints.instruments as instruments
    _HAS_OANDA = True
except ImportError:
    _HAS_OANDA = False
    warnings.warn("oandapyV20 not installed – Forex live feed unavailable.", stacklevel=1)


# ══════════════════════════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════════════════════════

# How many bars to fetch per timeframe
LIMIT: dict[str, int] = {"4h": 200, "1h": 500, "5m": 1000}

# Rolling-window size for swing highs/lows
SWING_N: dict[str, int] = {"4h": 10, "1h": 5, "5m": 5}

# ccxt timeframe strings
CCXT_TF: dict[str, str] = {"4h": "4h", "1h": "1h", "5m": "5m"}

# OANDA granularity codes
OANDA_TF: dict[str, str] = {"4h": "H4", "1h": "H1", "5m": "M5"}

# Local CSV suffix map  (matches gold_clean_data naming convention)
LOCAL_TF_SUFFIX: dict[str, str] = {"4h": "4h", "1h": "1h", "5m": "5m"}

# Session windows in UTC hours (half-open intervals [start, end))
SESSIONS: list[tuple[str, float, float]] = [
    ("london",   7.0,  10.0),
    ("newyork", 13.0,  16.0),
    ("nypm",    18.5,  21.0),
]


# ══════════════════════════════════════════════════════════════════════════════
#  Session labelling
# ══════════════════════════════════════════════════════════════════════════════

def _add_session_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a 'session' column using fully vectorized UTC-hour arithmetic.

    Replaces the previous .map(lambda) approach which iterated row-by-row
    and caused KeyboardInterrupt / timeout on large DataFrames (e.g. 1.4M 5M bars).
    This version computes the fractional UTC hour as a numpy array in one shot.
    """
    df = df.copy()
    # Fractional hour in UTC for every bar — one numpy op, no Python loop
    hour = df.index.hour + df.index.minute / 60.0  # numpy array

    # Default to "off", then overwrite with session windows (last write wins,
    # but windows are non-overlapping so order doesn't matter)
    session = pd.array(["off"] * len(df), dtype="object")
    for name, start, end in SESSIONS:
        mask = (hour >= start) & (hour < end)
        session[mask] = name

    df["session"] = session
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Swing high / low
# ══════════════════════════════════════════════════════════════════════════════

def _add_swings(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Add 'swing_high' and 'swing_low' columns.

    swing_high[i] = rolling max of 'high'  over the previous n bars (inclusive).
    swing_low[i]  = rolling min of 'low'   over the previous n bars (inclusive).
    """
    df = df.copy()
    df["swing_high"] = df["high"].rolling(n, min_periods=1).max()
    df["swing_low"]  = df["low"].rolling(n, min_periods=1).min()
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Raw OHLCV builders
# ══════════════════════════════════════════════════════════════════════════════

def _ohlcv_columns() -> list[str]:
    return ["timestamp", "open", "high", "low", "close", "volume"]


def _build_df(rows: list[list], ts_unit: str = "ms") -> pd.DataFrame:
    """Convert a list of [ts, o, h, l, c, v] rows into a tidy DataFrame."""
    df = pd.DataFrame(rows, columns=_ohlcv_columns())
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit=ts_unit, utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)
    return df


# ── ccxt (crypto) ─────────────────────────────────────────────────────────────

def _fetch_ccxt(symbol: str, exchange, tf: str, limit: int) -> pd.DataFrame:
    """Fetch OHLCV via a ccxt exchange object."""
    if not _HAS_CCXT:
        raise ImportError("ccxt is required for crypto live feed.")
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=CCXT_TF[tf], limit=limit)
    return _build_df(ohlcv, ts_unit="ms")


# ── oandapyV20 (Forex) ────────────────────────────────────────────────────────

def _fetch_oanda(symbol: str, client, tf: str, limit: int) -> pd.DataFrame:
    """
    Fetch OHLCV via oandapyV20.

    symbol  – OANDA instrument format, e.g. "EUR_USD"
    client  – oandapyV20.API instance
    """
    if not _HAS_OANDA:
        raise ImportError("oandapyV20 is required for Forex live feed.")

    params = {"granularity": OANDA_TF[tf], "count": limit, "price": "M"}
    req = instruments.InstrumentsCandles(instrument=symbol, params=params)
    client.request(req)

    rows = []
    for candle in req.response["candles"]:
        if not candle["complete"]:
            continue
        ts = pd.Timestamp(candle["time"]).value // 1_000_000  # → ms
        m  = candle["mid"]
        rows.append([
            ts,
            float(m["o"]),
            float(m["h"]),
            float(m["l"]),
            float(m["c"]),
            float(candle.get("volume", 0)),
        ])
    return _build_df(rows, ts_unit="ms")


# ── local CSV (back-test / offline) ───────────────────────────────────────────

def _fetch_local(tf: str, local_dir: str, local_prefix: str) -> pd.DataFrame:
    """
    Load a pre-downloaded CSV file.

    Expected filename pattern:
        {local_prefix}_{tf}_data_clean.csv   (e.g. XAU_4h_data_clean.csv)

    The CSV must have at minimum columns: open, high, low, close, volume
    and either a 'timestamp' column or a DatetimeIndex.
    """
    suffix = LOCAL_TF_SUFFIX[tf]
    candidates = [
        Path(local_dir) / f"{local_prefix}_{suffix}_data_clean.csv",
        Path(local_dir) / f"{local_prefix}_{suffix}.csv",
        Path(local_dir) / f"{local_prefix.lower()}_{suffix}.csv",
    ]
    path: Optional[Path] = None
    for c in candidates:
        if c.exists():
            path = c
            break

    if path is None:
        checked = ", ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            f"[data_feed] No local CSV found for timeframe '{tf}'. Checked: {checked}"
        )

    df = pd.read_csv(path)

    # Normalise column names to lowercase
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse timestamp / index
    ts_col = next(
        (c for c in df.columns if c in ("timestamp", "time", "date", "datetime")),
        None,
    )
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col], utc=True, format="ISO8601", errors="coerce")
        df.set_index(ts_col, inplace=True)
        df.index.name = "timestamp"
    else:
        # Try treating the first column as the index
        df.index = pd.to_datetime(df.index, utc=True, format="ISO8601", errors="coerce")
        df.index.name = "timestamp"

    df.sort_index(inplace=True)

    # Ensure required columns exist
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[data_feed] CSV '{path}' missing columns: {missing}")

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ══════════════════════════════════════════════════════════════════════════════
#  Dispatcher – detect exchange type automatically
# ══════════════════════════════════════════════════════════════════════════════

def _detect_source(exchange) -> str:
    """Return 'ccxt', 'oanda', or raise."""
    if exchange is None:
        return "local"
    if _HAS_CCXT and isinstance(exchange, ccxt.Exchange):
        return "ccxt"
    if _HAS_OANDA and isinstance(exchange, oandapyV20.API):
        return "oanda"
    # Allow duck-typed objects (e.g. mocks in tests)
    cls_name = type(exchange).__name__.lower()
    if "oanda" in cls_name or "api" in cls_name:
        return "oanda"
    return "ccxt"


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def fetch_mtf(
    symbol: str,
    exchange=None,
    *,
    source: str = "auto",
    local_dir: str = "gold_clean_data",
    local_prefix: str = "XAU",
    limits: Optional[dict[str, int]] = None,
) -> dict[str, pd.DataFrame]:
    """
    Fetch multi-timeframe OHLCV data and return enriched DataFrames.

    Parameters
    ----------
    symbol      : Trading pair, e.g. "BTC/USDT" (ccxt) or "EUR_USD" (oanda).
    exchange    : ccxt exchange instance  OR  oandapyV20.API instance.
                  Pass None (or omit) for local CSV mode.
    source      : "auto" (default) | "ccxt" | "oanda" | "local"
    local_dir   : Directory containing local CSV files.
    local_prefix: Filename prefix for local CSV files (e.g. "XAU").
    limits      : Override default bar counts, e.g. {"5m": 2000}.

    Returns
    -------
    dict with keys "4h", "1h", "5m", each a pd.DataFrame with columns:
        open, high, low, close, volume, swing_high, swing_low, session
    """
    effective_limits = {**LIMIT, **(limits or {})}
    resolved_source = source if source != "auto" else _detect_source(exchange)

    result: dict[str, pd.DataFrame] = {}

    for tf in ("4h", "1h", "5m"):
        # ── 1. Fetch raw OHLCV ────────────────────────────────────────────
        if resolved_source == "ccxt":
            df = _fetch_ccxt(symbol, exchange, tf, effective_limits[tf])
        elif resolved_source == "oanda":
            df = _fetch_oanda(symbol, exchange, tf, effective_limits[tf])
        elif resolved_source == "local":
            df = _fetch_local(tf, local_dir, local_prefix)
        else:
            raise ValueError(
                f"Unknown source '{resolved_source}'. "
                "Use 'auto', 'ccxt', 'oanda', or 'local'."
            )

        # ── 2. Swing highs / lows ────────────────────────────────────────
        df = _add_swings(df, SWING_N[tf])

        # ── 3. Session labels ────────────────────────────────────────────
        df = _add_session_column(df)

        result[tf] = df

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Quick smoke-test  (python data_feed.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=== data_feed.py smoke test ===\n")

    # ── Local CSV (always works without API keys) ─────────────────────────
    script_dir = Path(__file__).parent
    local_path = script_dir / "gold_clean_data"

    if local_path.exists():
        print("[LOCAL] Loading XAU gold data from CSV …")
        try:
            data = fetch_mtf(
                "XAU/USD",
                source="local",
                local_dir=str(local_path),
                local_prefix="XAU",
            )
            for tf, df in data.items():
                print(
                    f"  [{tf}] shape={df.shape}  "
                    f"sessions={df['session'].value_counts().to_dict()}  "
                    f"swing_high_last={df['swing_high'].iloc[-1]:.5f}  "
                    f"swing_low_last={df['swing_low'].iloc[-1]:.5f}"
                )
                print(df.tail(3).to_string())
                print()
        except Exception as exc:
            print(f"  ERROR: {exc}")
    else:
        print(f"  Skipped local test – '{local_path}' not found.")

    # ── ccxt live (optional, skipped if ccxt not installed) ───────────────
    if _HAS_CCXT:
        print("\n[CCXT] Fetching BTC/USDT from Binance (no auth required) …")
        try:
            import ccxt as _ccxt
            ex = _ccxt.binance({"enableRateLimit": True})
            data = fetch_mtf("BTC/USDT", ex)
            for tf, df in data.items():
                print(
                    f"  [{tf}] shape={df.shape}  "
                    f"last_close={df['close'].iloc[-1]:.2f}  "
                    f"session={df['session'].iloc[-1]}"
                )
        except Exception as exc:
            print(f"  ERROR: {exc}")
    else:
        print("\n[CCXT] Skipped – ccxt not installed.")

