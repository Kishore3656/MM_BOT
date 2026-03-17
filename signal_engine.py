"""
signal_engine.py
================
Master ICT A+ signal generator — integrates all upstream modules.

Upstream column requirements
-----------------------------
data_feed       : session
market_structure: structure, last_structure_high, last_structure_low, choch, bos
dealing_range   : discount, premium, ote_low, ote_high, ote_short_low, ote_short_high
liquidity       : liquidity_sweep, sweep_dir, bsl_level, ssl_level
pd_arrays       : fvg_bull_active, fvg_bear_active, ob_bull_active, ob_bear_active, atr14

A+ LONG  (signal =  1): all 7 conditions + RR ≥ 2.0
A+ SHORT (signal = -1): all 7 mirrored conditions + RR ≥ 2.0
No trade (signal =  0): fewer than 7 conditions or RR too low

Per-signal output
-----------------
  signal          : 1 | -1 | 0
  conditions_met  : int (0–7)
  entry_price     : float  (close of signal bar)
  stop_loss       : float
  tp1             : float  (1 : 1  partial target)
  tp2             : float  (3 : 1  or opposite liquidity target)
  rr_ratio        : float
  kill_zone       : str    (session name)
  confluence      : list[str]   (names of triggered conditions)

Entry points
------------
    from signal_engine import generate_signals, score_signal

    # ── vectorised backtest scan ──────────────────────────────────
    signals = generate_signals(mtf_data)   # mtf_data fully enriched

    # ── single-bar live evaluation ────────────────────────────────
    result = score_signal(mtf_data, idx=-1)

Smoke test
----------
    python signal_engine.py
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# ── Tunable parameters ────────────────────────────────────────────────────────
KILL_ZONES:      frozenset[str] = frozenset({"london", "newyork"})
CHOCH_LOOKBACK:  int   = 10     # bars to look back for a directional CHoCH
SL_ATR_BUFFER:   float = 0.10   # SL placed SL_ATR_BUFFER × atr14 beyond structure low/high
DEFAULT_RR_MULT: float = 3.0    # fallback TP2 multiple when no liquidity level exists
MIN_RR:          float = 2.0    # minimum RR to emit a signal

CONDITION_NAMES: list[str] = [
    "c1_htf_bias",
    "c2_kill_zone",
    "c3_liq_sweep",
    "c4_choch",
    "c5_zone",
    "c6_ote",
    "c7_pd_array",
]

# ── Required columns per DataFrame ───────────────────────────────────────────
_REQ_4H = ["structure"]
_REQ_5M = [
    "session",
    "structure", "last_structure_high", "last_structure_low", "choch",
    "discount", "premium",
    "ote_low", "ote_high", "ote_short_low", "ote_short_high",
    "liquidity_sweep", "sweep_dir", "bsl_level", "ssl_level",
    "fvg_bull_active", "fvg_bear_active", "ob_bull_active", "ob_bear_active",
    "atr14",
]


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _check_cols(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[signal_engine] '{label}' is missing columns: {missing}.\n"
            "Ensure all upstream add_* functions have been applied."
        )


def _align_htf(htf_df: pd.DataFrame,
               ltf_index: pd.DatetimeIndex,
               cols: list[str]) -> pd.DataFrame:
    """Forward-fill higher-timeframe columns onto lower-timeframe index."""
    sub = htf_df[cols]
    combined_idx = sub.index.union(ltf_index)
    return sub.reindex(combined_idx).ffill().reindex(ltf_index)


def _get_mtf_frame(mtf_data: dict[str, pd.DataFrame], *keys: str) -> pd.DataFrame | None:
    """Return the first available timeframe DataFrame without using DataFrame truthiness."""
    for key in keys:
        frame = mtf_data.get(key)
        if frame is not None:
            return frame
    return None


def _build_sl(df: pd.DataFrame, direction: int) -> pd.Series:
    """
    Stop-loss price with a small ATR buffer beyond the structural level.

    Long  : below last_structure_low  − buffer
    Short : above last_structure_high + buffer
    """
    buf = df["atr14"] * SL_ATR_BUFFER
    if direction == 1:
        return (df["last_structure_low"] - buf).clip(lower=0)
    return df["last_structure_high"] + buf


def _build_tp2(
    entry: pd.Series,
    risk: pd.Series,
    liq_target: pd.Series,
    direction: int,
) -> pd.Series:
    """TP2 = opposite liquidity level if valid, else DEFAULT_RR_MULT × risk."""
    fallback = (
        entry + DEFAULT_RR_MULT * risk if direction == 1
        else entry - DEFAULT_RR_MULT * risk
    )
    if direction == 1:
        valid = liq_target.notna() & (liq_target > entry)
    else:
        valid = liq_target.notna() & (liq_target < entry)
    return liq_target.where(valid, fallback)


# ══════════════════════════════════════════════════════════════════════════════
#  Core condition logic
# ══════════════════════════════════════════════════════════════════════════════

def _build_conditions(
    df_5m: pd.DataFrame,
    htf_structure: pd.Series,
    choch_lookback: int,
) -> dict[str, tuple[pd.Series, pd.Series]]:
    """
    Compute all 7 boolean conditions for long and short.

    Returns a dict mapping condition name → (long_bool, short_bool).
    All Series are aligned to df_5m.index.
    """
    close = df_5m["close"]

    # ── C1: HTF structural bias ───────────────────────────────────────────────
    c1_long  = htf_structure == "bullish"
    c1_short = htf_structure == "bearish"

    # ── C2: Active kill zone ──────────────────────────────────────────────────
    c2 = df_5m["session"].isin(KILL_ZONES)

    # ── C3: Directional liquidity sweep ──────────────────────────────────────
    sweep = df_5m["liquidity_sweep"].fillna(False)
    c3_long  = sweep & (df_5m["sweep_dir"] == "ssl")   # sellside swept → bullish
    c3_short = sweep & (df_5m["sweep_dir"] == "bsl")   # buyside  swept → bearish

    # ── C4: CHoCH in the correct direction within lookback window ─────────────
    # A bullish CHoCH = choch fired AND the previous bar was in bearish structure
    prev_struct = df_5m["structure"].shift(1).fillna("consolidation")
    choch       = df_5m["choch"].fillna(False)

    choch_bull = (choch & (prev_struct == "bearish")).astype(np.int8)
    choch_bear = (choch & (prev_struct == "bullish")).astype(np.int8)

    window = choch_lookback
    c4_long  = choch_bull.rolling(window, min_periods=1).max().astype(bool)
    c4_short = choch_bear.rolling(window, min_periods=1).max().astype(bool)

    # ── C5: Price zone alignment ──────────────────────────────────────────────
    c5_long  = df_5m["discount"].fillna(False).astype(bool)
    c5_short = df_5m["premium"].fillna(False).astype(bool)

    # ── C6: Price inside OTE band ─────────────────────────────────────────────
    c6_long  = (close >= df_5m["ote_low"])       & (close <= df_5m["ote_high"])
    c6_short = (close >= df_5m["ote_short_low"]) & (close <= df_5m["ote_short_high"])
    c6_long  = c6_long.fillna(False)
    c6_short = c6_short.fillna(False)

    # ── C7: Active PD array at current price ─────────────────────────────────
    c7_long  = (df_5m["fvg_bull_active"].fillna(False) |
                df_5m["ob_bull_active"].fillna(False))
    c7_short = (df_5m["fvg_bear_active"].fillna(False) |
                df_5m["ob_bear_active"].fillna(False))

    return {
        "c1_htf_bias":  (c1_long,  c1_short),
        "c2_kill_zone": (c2,        c2),
        "c3_liq_sweep": (c3_long,  c3_short),
        "c4_choch":     (c4_long,  c4_short),
        "c5_zone":      (c5_long,  c5_short),
        "c6_ote":       (c6_long,  c6_short),
        "c7_pd_array":  (c7_long,  c7_short),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public API – vectorised scan
# ══════════════════════════════════════════════════════════════════════════════

def generate_signals(
    mtf_data: dict[str, pd.DataFrame],
    min_rr: float = MIN_RR,
    choch_lookback: int = CHOCH_LOOKBACK,
) -> pd.DataFrame:
    """
    Scan all 5M bars and return every A+ signal that meets all 7 conditions
    and a minimum RR ratio.

    Parameters
    ----------
    mtf_data      : fully-enriched dict {"4h": df_4h, "5m": df_5m, ...}
    min_rr        : minimum risk-reward ratio to emit (default 2.0)
    choch_lookback: bars to look back for a directional CHoCH (default 10)

    Returns
    -------
    pd.DataFrame indexed by signal bar timestamp with columns:
        signal, conditions_met, entry_price, stop_loss, tp1, tp2,
        rr_ratio, kill_zone, confluence,
        c1_htf_bias … c7_pd_array
    """
    df_4h = _get_mtf_frame(mtf_data, "4h", "4H")
    df_5m = _get_mtf_frame(mtf_data, "5m", "5M")

    if df_4h is None:
        raise KeyError("[signal_engine] mtf_data must contain '4h' key.")
    if df_5m is None:
        raise KeyError("[signal_engine] mtf_data must contain '5m' key.")

    _check_cols(df_4h, _REQ_4H, "4h")
    _check_cols(df_5m, _REQ_5M, "5m")

    # Align 4H structure to 5M index
    htf_structure = _align_htf(df_4h, df_5m.index, ["structure"])["structure"]

    # Compute all conditions
    conds = _build_conditions(df_5m, htf_structure, choch_lookback)

    # Score each direction
    long_scores  = sum(conds[n][0].astype(np.int8) for n in CONDITION_NAMES)
    short_scores = sum(conds[n][1].astype(np.int8) for n in CONDITION_NAMES)

    long_signal  = long_scores  == 7
    short_signal = short_scores == 7
    any_signal   = long_signal | short_signal

    if not any_signal.any():
        return pd.DataFrame(columns=[
            "signal", "conditions_met", "entry_price", "stop_loss",
            "tp1", "tp2", "rr_ratio", "kill_zone", "confluence",
            *CONDITION_NAMES,
        ])

    close = df_5m["close"]

    # ── SL / TP for each direction ────────────────────────────────────────────
    sl_long  = _build_sl(df_5m, direction=1)
    sl_short = _build_sl(df_5m, direction=-1)

    risk_long  = (close - sl_long).clip(lower=1e-9)
    risk_short = (sl_short - close).clip(lower=1e-9)

    tp1_long  = close + risk_long
    tp1_short = close - risk_short

    tp2_long  = _build_tp2(close, risk_long,  df_5m["bsl_level"],  1)
    tp2_short = _build_tp2(close, risk_short, df_5m["ssl_level"], -1)

    rr_long  = (tp2_long  - close) / risk_long
    rr_short = (close - tp2_short) / risk_short

    # Apply RR filter
    long_valid  = long_signal  & (rr_long  >= min_rr)
    short_valid = short_signal & (rr_short >= min_rr)

    # Long takes priority if both fire on the same bar (rare)
    direction = pd.Series(0, index=df_5m.index, dtype=np.int8)
    direction[short_valid] = -1
    direction[long_valid]  =  1

    sig_mask = direction != 0
    if not sig_mask.any():
        return pd.DataFrame(columns=[
            "signal", "conditions_met", "entry_price", "stop_loss",
            "tp1", "tp2", "rr_ratio", "kill_zone", "confluence",
            *CONDITION_NAMES,
        ])

    # ── Assemble output ───────────────────────────────────────────────────────
    idx = df_5m.index[sig_mask]
    d   = direction[sig_mask]
    is_long = d == 1

    scores = pd.Series(
        np.where(is_long,
                 long_scores[sig_mask].values,
                 short_scores[sig_mask].values),
        index=idx,
    )

    entry_p = close[sig_mask]
    sl_p    = pd.Series(np.where(is_long, sl_long[sig_mask],  sl_short[sig_mask]),  index=idx)
    tp1_p   = pd.Series(np.where(is_long, tp1_long[sig_mask], tp1_short[sig_mask]), index=idx)
    tp2_p   = pd.Series(np.where(is_long, tp2_long[sig_mask], tp2_short[sig_mask]), index=idx)
    rr_p    = pd.Series(np.where(is_long, rr_long[sig_mask],  rr_short[sig_mask]),  index=idx)

    # Per-condition columns (True = fired for the chosen direction)
    cond_cols: dict[str, pd.Series] = {}
    for name in CONDITION_NAMES:
        long_c, short_c = conds[name]
        cond_cols[name] = pd.Series(
            np.where(is_long, long_c[sig_mask].values, short_c[sig_mask].values),
            index=idx,
            dtype=bool,
        )

    # Confluence list as comma-separated string
    confluence_str = pd.Series(
        [
            ",".join(n for n in CONDITION_NAMES if cond_cols[n].iloc[k])
            for k in range(len(idx))
        ],
        index=idx,
    )

    out = pd.DataFrame({
        "signal":          d.values,
        "conditions_met":  scores.values,
        "entry_price":     entry_p.values,
        "stop_loss":       sl_p.values,
        "tp1":             tp1_p.values,
        "tp2":             tp2_p.values,
        "rr_ratio":        rr_p.values,
        "kill_zone":       df_5m["session"][sig_mask].values,
        "confluence":      confluence_str.values,
        **{n: cond_cols[n].values for n in CONDITION_NAMES},
    }, index=idx)

    return out


# ══════════════════════════════════════════════════════════════════════════════
#  Public API – single-bar live evaluation
# ══════════════════════════════════════════════════════════════════════════════

def score_signal(
    mtf_data: dict[str, pd.DataFrame],
    idx: int = -1,
    min_rr: float = MIN_RR,
    choch_lookback: int = CHOCH_LOOKBACK,
) -> dict[str, Any]:
    """
    Evaluate the ICT A+ conditions for a single bar (default: last bar).

    Parameters
    ----------
    mtf_data      : fully-enriched dict {"4h": df, "5m": df, ...}
    idx           : integer position in the 5M DataFrame (default -1 = latest)
    min_rr        : minimum RR required to set signal ≠ 0

    Returns
    -------
    dict with keys: signal, conditions_met, entry_price, stop_loss, tp1, tp2,
                    rr_ratio, kill_zone, confluence, bar_time, c1..c7
    """
    df_4h = _get_mtf_frame(mtf_data, "4h", "4H")
    df_5m = _get_mtf_frame(mtf_data, "5m", "5M")

    _check_cols(df_4h, _REQ_4H, "4h")
    _check_cols(df_5m, _REQ_5M, "5m")

    bar_time = df_5m.index[idx]

    htf_structure = _align_htf(df_4h, df_5m.index, ["structure"])["structure"]
    conds = _build_conditions(df_5m, htf_structure, choch_lookback)

    # Evaluate at the target bar
    row: dict[str, bool] = {
        "long":  {},
        "short": {},
    }
    for name in CONDITION_NAMES:
        long_c, short_c = conds[name]
        row["long"][name]  = bool(long_c.iloc[idx])
        row["short"][name] = bool(short_c.iloc[idx])

    long_score  = sum(row["long"].values())
    short_score = sum(row["short"].values())

    # Determine direction
    if long_score == 7:
        direction = 1
        score     = long_score
        cond_dict = row["long"]
    elif short_score == 7:
        direction = -1
        score     = short_score
        cond_dict = row["short"]
    else:
        # Best-effort: return the higher-scoring direction
        if long_score >= short_score:
            direction = 0
            score     = long_score
            cond_dict = row["long"]
        else:
            direction = 0
            score     = short_score
            cond_dict = row["short"]

    close  = float(df_5m["close"].iloc[idx])
    atr14  = float(df_5m["atr14"].iloc[idx])
    buf    = atr14 * SL_ATR_BUFFER

    if direction >= 0:   # long or undecided
        sl     = max(float(df_5m["last_structure_low"].iloc[idx])  - buf, 0.0)
        risk   = max(close - sl, 1e-9)
        tp1    = close + risk
        liq    = df_5m["bsl_level"].iloc[idx]
        tp2    = float(liq) if pd.notna(liq) and float(liq) > close else close + DEFAULT_RR_MULT * risk
        rr     = (tp2 - close) / risk
    else:                # short
        sl     = float(df_5m["last_structure_high"].iloc[idx]) + buf
        risk   = max(sl - close, 1e-9)
        tp1    = close - risk
        liq    = df_5m["ssl_level"].iloc[idx]
        tp2    = float(liq) if pd.notna(liq) and float(liq) < close else close - DEFAULT_RR_MULT * risk
        rr     = (close - tp2) / risk

    if direction != 0 and rr < min_rr:
        direction = 0

    confluence = [n for n, v in cond_dict.items() if v]

    return {
        "signal":          direction,
        "conditions_met":  score,
        "entry_price":     round(close, 5),
        "stop_loss":       round(sl,    5),
        "tp1":             round(tp1,   5),
        "tp2":             round(tp2,   5),
        "rr_ratio":        round(rr,    3),
        "kill_zone":       str(df_5m["session"].iloc[idx]),
        "confluence":      confluence,
        "bar_time":        bar_time,
        **{n: v for n, v in cond_dict.items()},
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test  (python signal_engine.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    import time
    from pathlib import Path

    print("=== signal_engine.py smoke test ===\n")

    script_dir = Path(__file__).parent
    local_path = script_dir / "gold_clean_data"

    if not local_path.exists():
        print(f"Skipped — '{local_path}' not found.")
        sys.exit(0)

    try:
        from data_feed       import fetch_mtf
        from market_structure import add_market_structure
        from dealing_range   import compute_dealing_range
        from liquidity       import add_liquidity
        from pd_arrays       import add_pd_arrays
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

    print("Enriching all timeframes …")
    t0 = time.perf_counter()

    enriched: dict[str, pd.DataFrame] = {}
    for tf, df in mtf.items():
        df = add_market_structure(df, pivot_n=5)
        df = compute_dealing_range(df, window="session")
        df = add_liquidity(df, lookback=20)
        df, _, _ = add_pd_arrays(df, timeframe=tf)
        enriched[tf] = df

    enrich_time = time.perf_counter() - t0
    print(f"  enrichment done in {enrich_time:.1f}s")

    print("\nRunning generate_signals …")
    t1 = time.perf_counter()
    signals = generate_signals(enriched, min_rr=2.0)
    scan_time = time.perf_counter() - t1

    n_long  = int((signals["signal"] ==  1).sum())
    n_short = int((signals["signal"] == -1).sum())
    print(f"  scan done in {scan_time:.2f}s")
    print(f"  Total A+ signals : {len(signals)}  (long={n_long}, short={n_short})")

    if len(signals):
        print(f"\n  Average RR       : {signals['rr_ratio'].mean():.2f}")
        print(f"  Median RR        : {signals['rr_ratio'].median():.2f}")
        print(f"\n  Last 5 signals:")
        display_cols = ["signal", "conditions_met", "entry_price",
                        "stop_loss", "tp2", "rr_ratio", "kill_zone"]
        print(signals[display_cols].tail(5).to_string())

    print("\nRunning score_signal (last 5M bar) …")
    result = score_signal(enriched, idx=-1)
    print(f"  bar_time       : {result['bar_time']}")
    print(f"  signal         : {result['signal']}")
    print(f"  conditions_met : {result['conditions_met']}/7")
    print(f"  entry          : {result['entry_price']}")
    print(f"  sl             : {result['stop_loss']}")
    print(f"  tp1            : {result['tp1']}")
    print(f"  tp2            : {result['tp2']}")
    print(f"  rr             : {result['rr_ratio']}")
    print(f"  kill_zone      : {result['kill_zone']}")
    print(f"  confluence     : {result['confluence']}")
