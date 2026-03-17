"""
liquidity.py
============
ICT liquidity pool mapping, sweep detection, and Judas Swing flagging.

Adds to each OHLCV DataFrame:
  bsl_level       – nearest active Buyside Liquidity level (NaN if none)
  ssl_level       – nearest active Sellside Liquidity level (NaN if none)
  eqh_level       – Equal High cluster price (NaN if none)
  eql_level       – Equal Low cluster price (NaN if none)
  liquidity_sweep – True when a wick exceeds a liquidity level but closes back
  swept_level     – price level that was swept (NaN otherwise)
  sweep_dir       – "bsl" (high swept) | "ssl" (low swept) | None
  judas_swing     – True when sweep occurs in London kill zone AND price
                    reverses > 50% of the sweep candle body within 3 candles

Entry point
-----------
    from liquidity import add_liquidity
    df = add_liquidity(df, lookback=20)

Smoke test
----------
    python liquidity.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
EQH_TOL: float = 0.0015          # 0.15 % for equal-high / equal-low clustering
LONDON_START: float = 7.0        # UTC hour (inclusive)
LONDON_END: float   = 10.0       # UTC hour (exclusive) — London kill zone
JUDAS_REVERSAL_BARS: int = 3     # candles after sweep to check reversal
JUDAS_REVERSAL_PCT: float = 0.50 # body reversal threshold


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _fractional_hour(index: pd.DatetimeIndex) -> np.ndarray:
    """Return UTC fractional hour array (no Python loop)."""
    return index.hour + index.minute / 60.0


def _cluster_levels(
    prices: np.ndarray,
    tol: float,
) -> list[float]:
    """
    Group prices that are within `tol` (relative) of each other.
    Returns the mean of each cluster that has >= 2 members.
    Prices must be sorted ascending.
    """
    if len(prices) == 0:
        return []

    clusters: list[list[float]] = [[prices[0]]]
    for p in prices[1:]:
        if abs(p - clusters[-1][-1]) / max(abs(clusters[-1][-1]), 1e-9) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])

    return [float(np.mean(c)) for c in clusters if len(c) >= 2]


def _active_external_levels(
    pivot_vals: np.ndarray,
    close_vals: np.ndarray,
    lookback: int,
    side: str,
) -> np.ndarray:
    """Vectorized nearest active liquidity level over trailing windows."""
    n = len(close_vals)
    out = np.full(n, np.nan)
    if n <= lookback:
        return out

    pivot_windows = np.lib.stride_tricks.sliding_window_view(pivot_vals, lookback)[:-1]
    close_windows = np.lib.stride_tricks.sliding_window_view(close_vals, lookback)[:-1]
    current_close = close_vals[lookback:]

    if side == "high":
        reversed_close = close_windows[:, ::-1]
        suffix_extreme = np.maximum.accumulate(reversed_close, axis=1)[:, ::-1]
        future_close = np.concatenate(
            [suffix_extreme[:, 1:], np.full((len(pivot_windows), 1), -np.inf)],
            axis=1,
        )
        valid = (
            ~np.isnan(pivot_windows)
            & (future_close <= pivot_windows)
            & (pivot_windows > current_close[:, None])
        )
        masked = np.where(valid, pivot_windows, np.inf)
        nearest = masked.min(axis=1)
        out[lookback:] = np.where(np.isfinite(nearest), nearest, np.nan)
        return out

    if side == "low":
        reversed_close = close_windows[:, ::-1]
        suffix_extreme = np.minimum.accumulate(reversed_close, axis=1)[:, ::-1]
        future_close = np.concatenate(
            [suffix_extreme[:, 1:], np.full((len(pivot_windows), 1), np.inf)],
            axis=1,
        )
        valid = (
            ~np.isnan(pivot_windows)
            & (future_close >= pivot_windows)
            & (pivot_windows < current_close[:, None])
        )
        masked = np.where(valid, pivot_windows, -np.inf)
        nearest = masked.max(axis=1)
        out[lookback:] = np.where(np.isfinite(nearest), nearest, np.nan)
        return out

    raise ValueError("[liquidity] side must be 'high' or 'low'.")


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 – External liquidity detection
# ══════════════════════════════════════════════════════════════════════════════

def find_external_liquidity(
    df: pd.DataFrame,
    lookback: int = 20,
) -> pd.DataFrame:
    """
    Detect Buyside / Sellside liquidity pools and Equal Highs / Equal Lows.

    For each bar i, scans the preceding `lookback` bars for:
    - BSL: swing highs that have NOT been broken (close > level never occurred)
    - SSL: swing lows that have NOT been broken (close < level never occurred)
    - EQH: clusters of swing highs within EQH_TOL of each other
    - EQL: clusters of swing lows within the same tolerance

    "Swing high" / "swing low" here uses the `pivot_high` / `pivot_low` columns
    produced by market_structure.find_pivots().  Falls back to rolling-window
    highs/lows if those columns are absent.

    Adds columns
    ------------
    bsl_level : nearest active BSL price above current close (NaN if none)
    ssl_level : nearest active SSL price below current close (NaN if none)
    eqh_level : most recent EQH cluster mean price (NaN if none)
    eql_level : most recent EQL cluster mean price (NaN if none)
    """
    df = df.copy()
    n = len(df)

    close = df["close"].values

    # Use pivot columns if available, else rolling-window approximation
    if "pivot_high" in df.columns:
        ph_vals = df["pivot_high"].values      # NaN at non-pivot bars
        pl_vals = df["pivot_low"].values
    else:
        # Fallback: rolling(5) argmax / argmin as a lightweight pivot proxy
        ph_series = df["high"].rolling(5, center=True, min_periods=3).max()
        pl_series = df["low"].rolling(5, center=True, min_periods=3).min()
        ph_vals = np.where(df["high"].values == ph_series.values, df["high"].values, np.nan)
        pl_vals = np.where(df["low"].values  == pl_series.values, df["low"].values,  np.nan)

    bsl_arr = _active_external_levels(ph_vals, close, lookback, side="high")
    ssl_arr = _active_external_levels(pl_vals, close, lookback, side="low")
    eqh_arr = np.full(n, np.nan)
    eql_arr = np.full(n, np.nan)

    for i in range(lookback, n):
        window_slice = slice(i - lookback, i)   # lookback bars ending BEFORE bar i
        c_now = close[i]

        ph_window = ph_vals[window_slice]
        pl_window = pl_vals[window_slice]

        # ── BSL: swing highs in window NOT yet broken by a close above them ──
        # ── SSL: swing lows in window NOT yet broken by a close below them ──
        # ── EQH: clusters of swing highs within tolerance ───────────────────
        ph_present = np.sort(ph_window[~np.isnan(ph_window)])
        eq_highs = _cluster_levels(ph_present, EQH_TOL)
        if eq_highs:
            # Most recently relevant EQH above current close (or highest)
            above = [p for p in eq_highs if p > c_now]
            eqh_arr[i] = min(above) if above else eq_highs[-1]

        # ── EQL: clusters of swing lows within tolerance ─────────────────────
        pl_present = np.sort(pl_window[~np.isnan(pl_window)])
        eq_lows = _cluster_levels(pl_present, EQH_TOL)
        if eq_lows:
            below = [p for p in eq_lows if p < c_now]
            eql_arr[i] = max(below) if below else eq_lows[0]

    df["bsl_level"] = bsl_arr
    df["ssl_level"] = ssl_arr
    df["eqh_level"] = eqh_arr
    df["eql_level"] = eql_arr

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 – Liquidity sweep detection
# ══════════════════════════════════════════════════════════════════════════════

def detect_sweeps(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify candles where the wick pierces a liquidity level but the close
    returns back through that level (the classic ICT liquidity sweep / stop hunt).

    BSL sweep : high > bsl_level  AND  close < bsl_level  (wick above, close below)
    SSL sweep : low  < ssl_level  AND  close > ssl_level  (wick below, close above)

    Adds columns
    ------------
    liquidity_sweep : bool
    swept_level     : float price level swept (NaN if none)
    sweep_dir       : "bsl" | "ssl" | None (object dtype)
    """
    df = df.copy()

    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    bsl = df["bsl_level"]
    ssl = df["ssl_level"]

    bsl_sweep = high.gt(bsl) & close.lt(bsl) & bsl.notna()
    ssl_sweep = low.lt(ssl)  & close.gt(ssl) & ssl.notna()

    sweep      = bsl_sweep | ssl_sweep
    swept_lvl  = np.where(bsl_sweep, bsl, np.where(ssl_sweep, ssl, np.nan))
    sweep_dir_vals = np.where(bsl_sweep, "bsl", np.where(ssl_sweep, "ssl", None))

    df["liquidity_sweep"] = sweep
    df["swept_level"]     = swept_lvl.astype(float)
    df["sweep_dir"]       = sweep_dir_vals

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 – Judas Swing flag
# ══════════════════════════════════════════════════════════════════════════════

def detect_judas_swing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flag Judas Swings: a liquidity sweep that occurs during the London kill zone
    (07:00–10:00 UTC) AND is followed by a reversal of > 50% of the sweep
    candle's body within the next JUDAS_REVERSAL_BARS candles.

    Adds column
    -----------
    judas_swing : bool
    """
    df = df.copy()
    n = len(df)

    hour       = _fractional_hour(df.index)
    in_london  = (hour >= LONDON_START) & (hour < LONDON_END)
    is_sweep   = df["liquidity_sweep"].values
    sweep_dir  = df["sweep_dir"].values
    open_      = df["open"].values
    close_     = df["close"].values
    high_      = df["high"].values
    low_       = df["low"].values

    judas = np.zeros(n, dtype=bool)

    for i in range(n - JUDAS_REVERSAL_BARS):
        if not (is_sweep[i] and in_london[i]):
            continue

        body = abs(close_[i] - open_[i])
        if body < 1e-9:
            continue  # doji — skip

        # Determine reversal direction and threshold
        if sweep_dir[i] == "bsl":
            # Swept buyside → bearish reversal expected
            # Reversal = price moves DOWN by > 50% of sweep body within next N bars
            reversal_target = close_[i] - body * JUDAS_REVERSAL_PCT
            future_lows = low_[i + 1 : i + 1 + JUDAS_REVERSAL_BARS]
            if np.any(future_lows <= reversal_target):
                judas[i] = True

        elif sweep_dir[i] == "ssl":
            # Swept sellside → bullish reversal expected
            reversal_target = close_[i] + body * JUDAS_REVERSAL_PCT
            future_highs = high_[i + 1 : i + 1 + JUDAS_REVERSAL_BARS]
            if np.any(future_highs >= reversal_target):
                judas[i] = True

    df["judas_swing"] = judas
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def add_liquidity(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """
    Full liquidity enrichment pipeline for a single-timeframe DataFrame.

    Parameters
    ----------
    df       : OHLCV DataFrame with UTC DatetimeIndex.
               Optionally has pivot_high / pivot_low from market_structure.py.
    lookback : bars to scan for liquidity levels (default 20)

    Returns
    -------
    DataFrame with added columns:
        bsl_level, ssl_level, eqh_level, eql_level,
        liquidity_sweep, swept_level, sweep_dir, judas_swing
    """
    df = find_external_liquidity(df, lookback=lookback)
    df = detect_sweeps(df)
    df = detect_judas_swing(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test  (python liquidity.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=== liquidity.py smoke test ===\n")

    script_dir = Path(__file__).parent
    local_path = script_dir / "gold_clean_data"

    if not local_path.exists():
        print(f"Skipped — '{local_path}' not found.")
        sys.exit(0)

    try:
        from data_feed import fetch_mtf
        from market_structure import add_market_structure
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Loading XAU gold data …")
    mtf = fetch_mtf(
        "XAU/USD",
        source="local",
        local_dir=str(local_path),
        local_prefix="XAU",
    )

    for tf, df in mtf.items():
        print(f"\n[{tf}] enriching …")
        df = add_market_structure(df, pivot_n=5)
        df = add_liquidity(df, lookback=20)

        n_rows   = len(df)
        n_bsl    = df["bsl_level"].notna().sum()
        n_ssl    = df["ssl_level"].notna().sum()
        n_eqh    = df["eqh_level"].notna().sum()
        n_eql    = df["eql_level"].notna().sum()
        n_sweep  = int(df["liquidity_sweep"].sum())
        n_bsl_sw = int((df["sweep_dir"] == "bsl").sum())
        n_ssl_sw = int((df["sweep_dir"] == "ssl").sum())
        n_judas  = int(df["judas_swing"].sum())

        print(f"  rows={n_rows:,}")
        print(f"  BSL levels active={n_bsl:,}  SSL levels active={n_ssl:,}")
        print(f"  EQH={n_eqh:,}  EQL={n_eql:,}")
        print(f"  Sweeps total={n_sweep:,}  (BSL swept={n_bsl_sw:,} / SSL swept={n_ssl_sw:,})")
        print(f"  Judas Swings={n_judas:,}")

        sweep_rows = df[df["liquidity_sweep"]].tail(3)
        if len(sweep_rows):
            cols = ["high", "low", "close", "sweep_dir",
                    "swept_level", "judas_swing", "bsl_level", "ssl_level"]
            print(sweep_rows[cols].to_string())
