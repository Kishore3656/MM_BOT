"""
market_structure.py
===================
ICT market structure: pivot detection, swing classification, BOS, CHoCH.

Adds to each OHLCV DataFrame:
  pivot_high           – confirmed pivot high price (NaN elsewhere)
  pivot_low            – confirmed pivot low price (NaN elsewhere)
  swing_type           – "HH" | "HL" | "LH" | "LL" | "none"
  structure            – "bullish" | "bearish" | "consolidation"
  last_structure_high  – most recent pivot high, forward-filled
  last_structure_low   – most recent pivot low, forward-filled
  bos                  – True when close breaks structural level in trend direction
  choch                – True when close breaks structural level against trend

Entry point
-----------
    from market_structure import add_market_structure
    df = add_market_structure(df, pivot_n=5)

Smoke test
----------
    python market_structure.py
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

try:
    from scipy.signal import argrelmax, argrelmin
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False
    warnings.warn("scipy not installed – pivot detection unavailable.", stacklevel=1)

# Equal high / equal low tolerance (relative)
EQ_TOL: float = 0.001  # 0.1 %


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 – Pivot detection (bidirectional)
# ══════════════════════════════════════════════════════════════════════════════

def find_pivots(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """
    Identify swing highs/lows using scipy argrelmax / argrelmin.

    n bars are required on EACH side of the candidate bar for confirmation.

    Adds columns
    ------------
    pivot_high : price at the pivot high, NaN elsewhere
    pivot_low  : price at the pivot low,  NaN elsewhere
    """
    if not _HAS_SCIPY:
        raise ImportError("scipy is required for pivot detection.")

    df = df.copy()
    high = df["high"].values
    low  = df["low"].values

    ph_idx = argrelmax(high, order=n)[0]
    pl_idx = argrelmin(low,  order=n)[0]

    df["pivot_high"] = np.nan
    df["pivot_low"]  = np.nan

    col_ph = df.columns.get_loc("pivot_high")
    col_pl = df.columns.get_loc("pivot_low")

    df.iloc[ph_idx, col_ph] = high[ph_idx]
    df.iloc[pl_idx, col_pl] = low[pl_idx]

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 – Swing label & structure state (fully vectorized)
# ══════════════════════════════════════════════════════════════════════════════

def classify_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classify each pivot as HH / LH (highs) or HL / LL (lows).
    Derive bullish / bearish / consolidation structure from the most recent
    classified pivot pair.

    Adds columns
    ------------
    swing_type           : "HH" | "HL" | "LH" | "LL" | "none"
    structure            : "bullish" | "bearish" | "consolidation"
    last_structure_high  : forward-filled price of most recent pivot high
    last_structure_low   : forward-filled price of most recent pivot low
    """
    df = df.copy()

    # ── Classify pivot highs ──────────────────────────────────────────────────
    ph = df["pivot_high"].dropna()
    if len(ph) >= 2:
        prev_ph  = ph.shift(1)
        tol_ph   = prev_ph.abs() * EQ_TOL
        ph_label = pd.Series("none", index=ph.index, dtype=object)
        ph_label[ph > prev_ph + tol_ph] = "HH"
        ph_label[ph < prev_ph - tol_ph] = "LH"
    else:
        ph_label = pd.Series("none", index=ph.index, dtype=object)

    # ── Classify pivot lows ───────────────────────────────────────────────────
    pl = df["pivot_low"].dropna()
    if len(pl) >= 2:
        prev_pl  = pl.shift(1)
        tol_pl   = prev_pl.abs() * EQ_TOL
        pl_label = pd.Series("none", index=pl.index, dtype=object)
        pl_label[pl > prev_pl + tol_pl] = "HL"
        pl_label[pl < prev_pl - tol_pl] = "LL"
    else:
        pl_label = pd.Series("none", index=pl.index, dtype=object)

    # ── Merge into swing_type (high label has priority on same-bar conflict) ──
    swing_type = pd.Series("none", index=df.index, dtype=object)

    labeled_ph = ph_label[ph_label != "none"]
    if len(labeled_ph):
        swing_type.loc[labeled_ph.index] = labeled_ph

    labeled_pl = pl_label[pl_label != "none"]
    if len(labeled_pl):
        no_high = swing_type.loc[labeled_pl.index] == "none"
        swing_type.loc[labeled_pl.index[no_high]] = labeled_pl[no_high]

    # ── Forward-fill last classified label per type ───────────────────────────
    full_ph_label = ph_label.reindex(df.index)
    full_pl_label = pl_label.reindex(df.index)

    last_high_label = full_ph_label.where(full_ph_label != "none").ffill()
    last_low_label  = full_pl_label.where(full_pl_label != "none").ffill()

    # ── Structure determination ───────────────────────────────────────────────
    bull = (last_high_label == "HH") & (last_low_label == "HL")
    bear = (last_high_label == "LH") & (last_low_label == "LL")

    structure = pd.Series("consolidation", index=df.index, dtype=object)
    structure[bull] = "bullish"
    structure[bear] = "bearish"

    # ── Structural price levels (forward-filled) ──────────────────────────────
    last_sh = df["pivot_high"].ffill()
    last_sl = df["pivot_low"].ffill()

    df["swing_type"]          = swing_type
    df["structure"]           = structure
    df["last_structure_high"] = last_sh
    df["last_structure_low"]  = last_sl

    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 – BOS (Break of Structure) — with-trend
# ══════════════════════════════════════════════════════════════════════════════

def detect_bos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close breaks the previous bar's structural level IN the trend direction.

    bullish BOS : close > last_structure_high[t-1]
    bearish BOS : close < last_structure_low[t-1]

    Adds column
    -----------
    bos : bool
    """
    df    = df.copy()
    close = df["close"]
    prev_sh = df["last_structure_high"].shift(1)
    prev_sl = df["last_structure_low"].shift(1)

    bos = (
        ((df["structure"] == "bullish") & (close > prev_sh)) |
        ((df["structure"] == "bearish") & (close < prev_sl))
    )
    df["bos"] = bos.fillna(False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 – CHoCH (Change of Character) — counter-trend
# ══════════════════════════════════════════════════════════════════════════════

def detect_choch(df: pd.DataFrame) -> pd.DataFrame:
    """
    Close breaks the previous bar's structural level AGAINST the trend.

    bullish CHoCH : was bullish, close < last_structure_low[t-1]
    bearish CHoCH : was bearish, close > last_structure_high[t-1]

    Adds column
    -----------
    choch : bool
    """
    df          = df.copy()
    close       = df["close"]
    prev_struct = df["structure"].shift(1).fillna("consolidation")
    prev_sh     = df["last_structure_high"].shift(1)
    prev_sl     = df["last_structure_low"].shift(1)

    choch = (
        ((prev_struct == "bullish") & (close < prev_sl)) |
        ((prev_struct == "bearish") & (close > prev_sh))
    )
    df["choch"] = choch.fillna(False)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════════════

def add_market_structure(df: pd.DataFrame, pivot_n: int = 5) -> pd.DataFrame:
    """
    Enrich a single-timeframe OHLCV DataFrame with ICT market structure.

    Parameters
    ----------
    df      : DataFrame with open, high, low, close columns and UTC DatetimeIndex
    pivot_n : bars required on each side to confirm a pivot (default 5)

    Returns
    -------
    DataFrame with added columns:
        pivot_high, pivot_low,
        swing_type, structure, last_structure_high, last_structure_low,
        bos, choch
    """
    df = find_pivots(df, n=pivot_n)
    df = classify_structure(df)
    df = detect_bos(df)
    df = detect_choch(df)
    return df


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test  (python market_structure.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=== market_structure.py smoke test ===\n")

    script_dir = Path(__file__).parent
    local_path = script_dir / "gold_clean_data"

    if not local_path.exists():
        print(f"Skipped — '{local_path}' not found.")
        sys.exit(0)

    try:
        from data_feed import fetch_mtf
    except ImportError:
        print("ERROR: data_feed.py not found in the same directory.")
        sys.exit(1)

    print("Loading XAU gold data …")
    mtf = fetch_mtf(
        "XAU/USD",
        source="local",
        local_dir=str(local_path),
        local_prefix="XAU",
    )

    for tf, df in mtf.items():
        print(f"\n[{tf}] pivot_n=5 …")
        df_ms = add_market_structure(df, pivot_n=5)

        n          = len(df_ms)
        n_ph       = df_ms["pivot_high"].notna().sum()
        n_pl       = df_ms["pivot_low"].notna().sum()
        sw_counts  = df_ms["swing_type"].value_counts().to_dict()
        st_counts  = df_ms["structure"].value_counts().to_dict()
        n_bos      = int(df_ms["bos"].sum())
        n_choch    = int(df_ms["choch"].sum())

        print(f"  rows={n:,}  pivot_highs={n_ph:,}  pivot_lows={n_pl:,}")
        print(f"  swing_type : {sw_counts}")
        print(f"  structure  : {st_counts}")
        print(f"  BOS={n_bos:,}  CHoCH={n_choch:,}")

        # Last 3 pivot bars
        recent = df_ms[df_ms["swing_type"] != "none"].tail(3)
        if len(recent):
            print(recent[["open","high","low","close","swing_type",
                           "structure","bos","choch"]].to_string())
