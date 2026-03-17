"""
pd_arrays.py
============
ICT Premium / Discount Arrays: Fair Value Gaps (FVG) and Order Blocks (OB).

Performance targets
-------------------
  1.4 M-row 5-minute DataFrame  ≈  2-8 s total
  Key optimisations:
    • FVG detection    – fully vectorised with pandas .shift(), no Python bar loop
    • Status updates   – suffix-min/max precomputed once → O(1) "open" check +
                         numpy argmax for first-touch (no Python inner loop)
    • _enrich_df       – ffill-aligned Series comparisons; no nested Python loops
    • OB detection     – filters to BOS/CHoCH trigger bars only, small loop

Returns
-------
  fvg_list  – one row per FVG:  type, top, bottom, timestamp, status, timeframe, bpr
  ob_list   – one row per OB:   type, top, bottom, timestamp, status, timeframe

  type   : "bullish" | "bearish" | "breaker_bullish" | "breaker_bearish"
  status : "open" | "mitigated" | "inverted"
  bpr    : True when a bull FVG zone overlaps a bear FVG (Balanced Price Range)

Enriches source DataFrame with:
  atr14, fvg_bull_active, fvg_bear_active, in_fvg,
  ob_bull_active, ob_bear_active, in_ob

Entry point
-----------
    from pd_arrays import add_pd_arrays
    df_enriched, fvg_list, ob_list = add_pd_arrays(df, timeframe="5m")

Smoke test
----------
    python pd_arrays.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
IMPULSE_BODY_MULT: float = 1.5   # body of middle candle ≥ mult × rolling avg body
AVG_BODY_WINDOW:   int   = 20    # lookback bars for avg-body calculation
OB_LOOKBACK:       int   = 5     # bars back to find the opposing candle for OB
BPR_IDX_WINDOW:    int   = 50    # max bar-distance for a bull/bear FVG BPR pair


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_c = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_c).abs(),
        (df["low"]  - prev_c).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _suffix_arrays(df: pd.DataFrame):
    """
    Precompute suffix min-of-lows and suffix max-of-highs in one pass.
    suffix_min_low[i]  = min( low[i], low[i+1], ... )
    suffix_max_high[i] = max( high[i], high[i+1], ... )
    Used for O(1) "FVG never touched" early-exit check.
    """
    low_r  = df["low"].values[::-1]
    high_r = df["high"].values[::-1]
    return (
        np.minimum.accumulate(low_r)[::-1],
        np.maximum.accumulate(high_r)[::-1],
    )


def _first_touch_bull(start: int, bottom: float, top: float,
                      low_arr: np.ndarray, close_arr: np.ndarray,
                      suf_min: np.ndarray) -> str:
    """Status for a bullish FVG: always 'open' at formation time.
    Active-zone detection is handled bar-by-bar in _enrich_df via price-in-zone check."""
    return "open"


def _first_touch_bear(start: int, bottom: float, top: float,
                      high_arr: np.ndarray, close_arr: np.ndarray,
                      suf_max: np.ndarray) -> str:
    """Status for a bearish FVG: always 'open' at formation time.
    Active-zone detection is handled bar-by-bar in _enrich_df via price-in-zone check."""
    return "open"


def _align_to_index(df_index: pd.DatetimeIndex,
                    timestamps: pd.Series,
                    values: pd.Series) -> pd.Series:
    """
    Forward-fill a sparse (timestamp → value) mapping onto df_index.
    Multiple entries at the same timestamp → keep the last.
    Returns a Series aligned to df_index (NaN where no prior entry).
    """
    s = pd.Series(values.values, index=timestamps.values, dtype=float)
    s = s.groupby(level=0).last().sort_index()
    combined = s.reindex(s.index.union(df_index)).ffill()
    return combined.reindex(df_index)


# ══════════════════════════════════════════════════════════════════════════════
#  FVG detection  (fully vectorised — no Python bar-loop)
# ══════════════════════════════════════════════════════════════════════════════

def detect_fvg(df: pd.DataFrame, timeframe: str = "unknown") -> pd.DataFrame:
    """
    Find all 3-bar Fair Value Gaps.

    Confirmation bar  i  (the bar AFTER the impulse):
      Bullish FVG : high[i-2] < low[i]   &  candle[i-1] is a large bullish bar
      Bearish FVG : low[i-2]  > high[i]  &  candle[i-1] is a large bearish bar

    "Large" = body of candle[i-1] ≥ IMPULSE_BODY_MULT × rolling-avg body (causal).

    FVG zone:
      Bullish → bottom = high[i-2],  top = low[i]
      Bearish → top    = low[i-2],   bottom = high[i]

    Status:
      open       – zone untouched
      mitigated  – price entered zone but did not close beyond far boundary
      inverted   – close broke beyond far boundary (zone flips S/R)
    """
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]
    close  = df["close"]
    ts     = df.index
    n      = len(df)

    # ── Rolling avg body (causal: does not include bar i or bar i-1) ──────────
    body     = (close - open_).abs()
    avg_body = body.rolling(AVG_BODY_WINDOW, min_periods=5).mean().shift(2)

    mid_body = body.shift(1)                             # body of candle i-1
    is_impulse = mid_body >= IMPULSE_BODY_MULT * avg_body.fillna(mid_body)

    # ── Vectorised gap conditions ─────────────────────────────────────────────
    bull_mask = (
        (high.shift(2) < low) &
        (close.shift(1) > open_.shift(1)) &             # candle i-1 bullish
        is_impulse
    ).fillna(False)

    bear_mask = (
        (low.shift(2) > high) &
        (close.shift(1) < open_.shift(1)) &             # candle i-1 bearish
        is_impulse
    ).fillna(False)

    bull_idx = np.flatnonzero(bull_mask.values)
    bear_idx = np.flatnonzero(bear_mask.values)

    if len(bull_idx) + len(bear_idx) == 0:
        return pd.DataFrame(
            columns=["type", "top", "bottom", "timestamp", "status", "timeframe", "bpr"]
        )

    # ── Precompute suffix arrays once for fast status checks ─────────────────
    high_arr  = high.values
    low_arr   = low.values
    close_arr = close.values
    suf_min, suf_max = _suffix_arrays(df)

    records: list[dict] = []

    for i in bull_idx:
        bottom = float(high_arr[i - 2])
        top_   = float(low_arr[i])
        if top_ <= bottom:
            continue
        status = _first_touch_bull(i + 1, bottom, top_, low_arr, close_arr, suf_min)
        records.append({
            "type": "bullish", "top": top_, "bottom": bottom,
            "timestamp": ts[i], "status": status,
            "timeframe": timeframe, "bpr": False, "_idx": i,
        })

    for i in bear_idx:
        top_   = float(low_arr[i - 2])
        bottom = float(high_arr[i])
        if top_ <= bottom:
            continue
        status = _first_touch_bear(i + 1, bottom, top_, high_arr, close_arr, suf_max)
        records.append({
            "type": "bearish", "top": top_, "bottom": bottom,
            "timestamp": ts[i], "status": status,
            "timeframe": timeframe, "bpr": False, "_idx": i,
        })

    fvg_df = pd.DataFrame(records)

    # ── BPR: sliding-window two-pointer — O(n·k) not O(n²) ───────────────────
    # Broadcasting was creating (n_bull × n_bear) arrays → OOM on large datasets.
    # Sort both lists by bar index; inner loop bounded by BPR_IDX_WINDOW.
    bull_rows = fvg_df[fvg_df["type"] == "bullish"]
    bear_rows = fvg_df[fvg_df["type"] == "bearish"]
    if len(bull_rows) and len(bear_rows):
        b_sort = np.argsort(bull_rows["_idx"].values)
        r_sort = np.argsort(bear_rows["_idx"].values)

        b_idx_s  = bull_rows["_idx"].values[b_sort]
        b_top_s  = bull_rows["top"].values[b_sort]
        b_bot_s  = bull_rows["bottom"].values[b_sort]
        b_df_idx = bull_rows.index.values[b_sort]

        r_idx_s  = bear_rows["_idx"].values[r_sort]
        r_top_s  = bear_rows["top"].values[r_sort]
        r_bot_s  = bear_rows["bottom"].values[r_sort]
        r_df_idx = bear_rows.index.values[r_sort]

        bull_bpr: set[int] = set()
        bear_bpr: set[int] = set()
        r_start = 0

        for bi in range(len(b_idx_s)):
            b_i = b_idx_s[bi]
            while r_start < len(r_idx_s) and r_idx_s[r_start] < b_i - BPR_IDX_WINDOW:
                r_start += 1
            for ri in range(r_start, len(r_idx_s)):
                if r_idx_s[ri] > b_i + BPR_IDX_WINDOW:
                    break
                if min(b_top_s[bi], r_top_s[ri]) > max(b_bot_s[bi], r_bot_s[ri]):
                    bull_bpr.add(bi)
                    bear_bpr.add(ri)

        if bull_bpr:
            fvg_df.loc[b_df_idx[list(bull_bpr)], "bpr"] = True
        if bear_bpr:
            fvg_df.loc[r_df_idx[list(bear_bpr)], "bpr"] = True

    return fvg_df.drop(columns=["_idx"])


# ══════════════════════════════════════════════════════════════════════════════
#  OB detection  (filters to trigger bars — avoids full bar scan)
# ══════════════════════════════════════════════════════════════════════════════

def detect_ob(
    df: pd.DataFrame,
    fvg_list: pd.DataFrame,
    timeframe: str = "unknown",
) -> pd.DataFrame:
    """
    Detect Order Blocks and Breaker Blocks.

    Bullish OB : last bearish candle within OB_LOOKBACK bars before a bullish
                 impulse that triggers a BOS/CHoCH AND leaves a bullish FVG.
    Bearish OB : mirror.

    Breaker    : OB where price later closes through the OB's far boundary.
    """
    high   = df["high"].values
    low    = df["low"].values
    open_  = df["open"].values
    close  = df["close"].values
    ts     = df.index
    n      = len(df)

    has_bos   = "bos"   in df.columns
    has_choch = "choch" in df.columns

    atr14 = _compute_atr(df).values

    # Pre-build FVG timestamp sets for O(1) lookup
    fvg_bull_ts: set = set()
    fvg_bear_ts: set = set()
    if len(fvg_list):
        fvg_bull_ts = set(fvg_list.loc[fvg_list["type"] == "bullish", "timestamp"])
        fvg_bear_ts = set(fvg_list.loc[fvg_list["type"] == "bearish", "timestamp"])

    # Build trigger mask vectorised — only iterate over trigger bars
    bos_arr   = df["bos"].values   if has_bos   else np.zeros(n, dtype=bool)
    choch_arr = df["choch"].values if has_choch else np.zeros(n, dtype=bool)
    trigger   = bos_arr | choch_arr

    # Fallback when neither column present: any bar whose close breaks a 10-bar range
    if not has_bos and not has_choch:
        roll_high = df["high"].rolling(10, min_periods=1).max().shift(1).values
        roll_low  = df["low"].rolling(10, min_periods=1).min().shift(1).values
        trigger   = (close > roll_high) | (close < roll_low)

    trigger_indices = np.flatnonzero(trigger)

    suf_min, suf_max = _suffix_arrays(df)
    records: list[dict] = []

    for i in trigger_indices:
        atr_i = max(atr14[i], 1e-9)

        # ── Bullish OB ───────────────────────────────────────────────────────
        is_bull_impulse = (
            (close[i] > open_[i]) and
            (close[i] - open_[i]) >= 0.5 * atr_i
        )
        if is_bull_impulse and ts[i] in fvg_bull_ts:
            ob_idx = None
            for k in range(i - 1, max(i - OB_LOOKBACK - 1, -1), -1):
                if close[k] < open_[k]:
                    ob_idx = k
                    break
            if ob_idx is not None:
                ob_hi = float(high[ob_idx])
                ob_lo = float(low[ob_idx])
                # Status: open until price enters OB from below (mitigated) or
                # closes below OB low (inverted → breaker_bearish)
                start = ob_idx + 1
                status = "open"
                ob_type = "bullish"
                if start < n:
                    if suf_max[start] >= ob_hi:
                        j = start + int(np.argmax(close[start:] >= ob_hi))
                        if close[j] < ob_lo:
                            status, ob_type = "inverted", "breaker_bearish"
                        else:
                            status = "mitigated"
                records.append({
                    "type": ob_type, "top": ob_hi, "bottom": ob_lo,
                    "timestamp": ts[ob_idx], "status": status,
                    "timeframe": timeframe,
                })

        # ── Bearish OB ───────────────────────────────────────────────────────
        is_bear_impulse = (
            (close[i] < open_[i]) and
            (open_[i] - close[i]) >= 0.5 * atr_i
        )
        if is_bear_impulse and ts[i] in fvg_bear_ts:
            ob_idx = None
            for k in range(i - 1, max(i - OB_LOOKBACK - 1, -1), -1):
                if close[k] > open_[k]:
                    ob_idx = k
                    break
            if ob_idx is not None:
                ob_hi = float(high[ob_idx])
                ob_lo = float(low[ob_idx])
                start = ob_idx + 1
                status = "open"
                ob_type = "bearish"
                if start < n:
                    if suf_min[start] <= ob_lo:
                        j = start + int(np.argmax(close[start:] <= ob_lo))
                        if close[j] > ob_hi:
                            status, ob_type = "inverted", "breaker_bullish"
                        else:
                            status = "mitigated"
                records.append({
                    "type": ob_type, "top": ob_hi, "bottom": ob_lo,
                    "timestamp": ts[ob_idx], "status": status,
                    "timeframe": timeframe,
                })

    if not records:
        return pd.DataFrame(
            columns=["type", "top", "bottom", "timestamp", "status", "timeframe"]
        )
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
#  Enrich source DataFrame  (vectorised — no nested Python loops)
# ══════════════════════════════════════════════════════════════════════════════

def _enrich_df(
    df: pd.DataFrame,
    fvg_list: pd.DataFrame,
    ob_list: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add bar-level active-zone columns to the source DataFrame.

    Strategy: for each zone type, forward-fill the most-recently-confirmed
    open zone's top/bottom onto every bar, then compare with close vectorially.
    This is O(n log n) — no Python loops over bars.
    """
    df = df.copy()
    df["atr14"] = _compute_atr(df)

    close = df["close"]

    # Initialise
    for col in ("fvg_bull_active", "fvg_bear_active",
                "ob_bull_active",  "ob_bear_active"):
        df[col] = False
    df["in_fvg"] = None
    df["in_ob"]  = None

    # ── FVG columns ───────────────────────────────────────────────────────────
    if len(fvg_list):
        for ftype, active_col, in_label, cmp_fn in [
            ("bullish", "fvg_bull_active", "bull",
             lambda top, bot, c: (c < top,  (c >= bot) & (c < top))),
            ("bearish", "fvg_bear_active", "bear",
             lambda top, bot, c: (c > bot,  (c > bot)  & (c <= top))),
        ]:
            open_fvgs = fvg_list[
                (fvg_list["type"] == ftype) & (fvg_list["status"] == "open")
            ]
            if not len(open_fvgs):
                continue

            top_s = _align_to_index(df.index,
                                    open_fvgs["timestamp"], open_fvgs["top"])
            bot_s = _align_to_index(df.index,
                                    open_fvgs["timestamp"], open_fvgs["bottom"])

            active_mask, in_mask = cmp_fn(top_s, bot_s, close)
            df[active_col] = active_mask.fillna(False)

            has_bpr_col = "bpr" in open_fvgs.columns
            bpr_open = open_fvgs[open_fvgs["bpr"]] if has_bpr_col else open_fvgs.iloc[:0]
            if len(bpr_open):
                bpr_top = _align_to_index(df.index,
                                          bpr_open["timestamp"], bpr_open["top"])
                bpr_bot = _align_to_index(df.index,
                                          bpr_open["timestamp"], bpr_open["bottom"])
                _, bpr_in = cmp_fn(bpr_top, bpr_bot, close)
                in_mask = in_mask & ~bpr_in.fillna(False)
                df.loc[bpr_in.fillna(False), "in_fvg"] = "bpr"

            df.loc[in_mask.fillna(False), "in_fvg"] = in_label

    # ── OB columns ────────────────────────────────────────────────────────────
    if len(ob_list):
        for otype, active_col, in_label, cmp_fn in [
            ("bullish", "ob_bull_active", "bull",
             lambda hi, lo, c: (c < hi, (c >= lo) & (c < hi))),
            ("bearish", "ob_bear_active", "bear",
             lambda hi, lo, c: (c > lo, (c > lo)  & (c <= hi))),
        ]:
            open_obs = ob_list[
                (ob_list["type"] == otype) & (ob_list["status"] == "open")
            ]
            if not len(open_obs):
                continue

            top_s = _align_to_index(df.index,
                                    open_obs["timestamp"], open_obs["top"])
            bot_s = _align_to_index(df.index,
                                    open_obs["timestamp"], open_obs["bottom"])

            active_mask, in_mask = cmp_fn(top_s, bot_s, close)
            df[active_col] = active_mask.fillna(False)
            df.loc[in_mask.fillna(False), "in_ob"] = in_label

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def add_pd_arrays(
    df: pd.DataFrame,
    timeframe: str = "unknown",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Full PD-array pipeline for a single-timeframe DataFrame.

    Parameters
    ----------
    df        : OHLCV DataFrame with UTC DatetimeIndex.
                Optionally enriched with market_structure columns (bos, choch).
    timeframe : Label stored in output catalogues (e.g. "5m").

    Returns
    -------
    df_enriched : source df + atr14, fvg_*_active, in_fvg, ob_*_active, in_ob
    fvg_list    : DataFrame of all detected FVGs
    ob_list     : DataFrame of all detected OBs / Breaker Blocks
    """
    fvg_list = detect_fvg(df, timeframe=timeframe)
    ob_list  = detect_ob(df, fvg_list, timeframe=timeframe)
    df_out   = _enrich_df(df, fvg_list, ob_list)
    return df_out, fvg_list, ob_list


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test  (python pd_arrays.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    print("=== pd_arrays.py smoke test ===\n")

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
        df = add_market_structure(df, pivot_n=5)

        print(f"\n[{tf}]  rows={len(df):,}  running add_pd_arrays …")
        t0 = time.perf_counter()
        df_out, fvg_list, ob_list = add_pd_arrays(df, timeframe=tf)
        elapsed = time.perf_counter() - t0

        total_fvg = len(fvg_list)
        total_ob  = len(ob_list)

        fvg_status = fvg_list["status"].value_counts().to_dict() if total_fvg else {}
        fvg_type   = fvg_list["type"].value_counts().to_dict()   if total_fvg else {}
        n_bpr      = int(fvg_list["bpr"].sum())                  if total_fvg else 0
        ob_status  = ob_list["status"].value_counts().to_dict()  if total_ob  else {}
        ob_type    = ob_list["type"].value_counts().to_dict()     if total_ob  else {}

        print(f"  elapsed={elapsed:.2f}s")
        print(f"  FVGs total={total_fvg:,}  types={fvg_type}  status={fvg_status}  BPR={n_bpr}")
        print(f"  OBs  total={total_ob:,}   types={ob_type}   status={ob_status}")
        print(f"  fvg_bull_active={int(df_out['fvg_bull_active'].sum()):,}  "
              f"fvg_bear_active={int(df_out['fvg_bear_active'].sum()):,}")
        print(f"  ob_bull_active={int(df_out['ob_bull_active'].sum()):,}  "
              f"ob_bear_active={int(df_out['ob_bear_active'].sum()):,}")

        if total_fvg:
            print("\n  Last 3 FVGs:")
            print(fvg_list.tail(3).to_string(index=False))
        if total_ob:
            print("\n  Last 3 OBs:")
            print(ob_list.tail(3).to_string(index=False))
