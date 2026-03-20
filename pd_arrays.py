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
ZONE_LOOKBACK:     int   = 2000  # bars to look back for multi-zone bear active check


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

class _Fenwick:
    """Fenwick tree over counts; supports 'first 1 at/after idx' queries."""

    def __init__(self, n: int):
        self.n = int(n)
        self.bit = np.zeros(self.n + 1, dtype=np.int32)

    def add(self, idx0: int, delta: int = 1) -> None:
        i = int(idx0) + 1
        n = self.n
        bit = self.bit
        while i <= n:
            bit[i] += delta
            i += i & -i

    def sum(self, idx0: int) -> int:
        """Prefix sum over [0..idx0] inclusive."""
        i = int(idx0) + 1
        s = 0
        bit = self.bit
        while i > 0:
            s += int(bit[i])
            i -= i & -i
        return s

    def total(self) -> int:
        return self.sum(self.n - 1) if self.n else 0

    def find_by_order(self, k: int) -> int:
        """
        Return smallest idx such that prefix_sum(idx) >= k.
        Requires 1 <= k <= total().
        """
        idx = 0
        bit = self.bit
        # Largest power of two >= n
        step = 1 << (self.n.bit_length())
        while step:
            nxt = idx + step
            if nxt <= self.n and bit[nxt] < k:
                k -= int(bit[nxt])
                idx = nxt
            step >>= 1
        return idx  # 0-based (since idx is last < k in 1-based space)


def _first_touch_indices_threshold_leq(
    threshold: np.ndarray,
    start_idx: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    """
    For each query q: find the earliest j >= start_idx[q] such that series[j] <= threshold[q].
    Returns first_touch_idx (int), with n meaning "never touched".

    Offline algorithm:
      - sort bars by series ascending (activate bars with series <= current threshold)
      - sort queries by threshold ascending
      - Fenwick tree over activated bar indices; query next active >= start via order-statistic.
    """
    n = int(series.shape[0])
    out = np.full(len(threshold), n, dtype=np.int32)
    if n == 0 or len(threshold) == 0:
        return out

    bar_order = np.argsort(series, kind="mergesort")
    q_order = np.argsort(threshold, kind="mergesort")

    ft = _Fenwick(n)
    bi = 0
    total = 0

    for qi in q_order:
        thr = threshold[qi]
        while bi < n and series[bar_order[bi]] <= thr:
            ft.add(int(bar_order[bi]), 1)
            total += 1
            bi += 1

        s = int(start_idx[qi])
        if total == 0 or s >= n:
            continue

        before = ft.sum(s - 1) if s > 0 else 0
        if before >= total:
            continue

        j = ft.find_by_order(before + 1)
        out[qi] = int(j)

    return out


def _first_touch_indices_threshold_geq(
    threshold: np.ndarray,
    start_idx: np.ndarray,
    series: np.ndarray,
) -> np.ndarray:
    """
    For each query q: find the earliest j >= start_idx[q] such that series[j] >= threshold[q].
    Returns first_touch_idx (int), with n meaning "never touched".
    """
    # Convert >= into <= by negating both sides.
    return _first_touch_indices_threshold_leq(-threshold, start_idx, -series)


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


def _status_from_first_touch(
    zone_type: str,
    top: float,
    bottom: float,
    first_touch_idx: int,
    close_arr: np.ndarray,
    n: int,
) -> str:
    """
    End-of-dataset status classification based on the first touch bar.
    This is used for the zone catalogue (fvg_list / ob_list) only.

    Time-aware activeness at bar T is computed separately in _enrich_df.
    """
    if first_touch_idx >= n:
        return "open"
    c = float(close_arr[int(first_touch_idx)])
    if zone_type in ("bullish", "breaker_bullish"):
        return "inverted" if c < bottom else "mitigated"
    if zone_type in ("bearish", "breaker_bearish"):
        return "inverted" if c > top else "mitigated"
    return "mitigated"


def _timeaware_active_last_zone(
    n: int,
    created_idx: np.ndarray,
    first_touch_idx: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    For each bar t, track the most-recent zone that is still 'open as-of t':
      created_idx <= t < first_touch_idx

    Returns:
      last_top[t], last_bottom[t], any_active[t]
    """
    import heapq

    last_top = np.full(n, np.nan, dtype=float)
    last_bottom = np.full(n, np.nan, dtype=float)
    any_active = np.zeros(n, dtype=bool)

    if n == 0 or len(created_idx) == 0:
        return last_top, last_bottom, any_active

    created_idx = created_idx.astype(np.int32, copy=False)
    first_touch_idx = first_touch_idx.astype(np.int32, copy=False)

    # Group zones by created index for O(1) add.
    order = np.argsort(created_idx, kind="mergesort")
    ci_sorted = created_idx[order]
    ft_sorted = first_touch_idx[order]
    top_sorted = top[order]
    bot_sorted = bottom[order]

    # Max-heap keyed by created_idx (store negative for heapq) with lazy deletion by expiry.
    heap: list[tuple[int, int]] = []  # (-created_idx, ptr)

    ptr = 0
    m = len(ci_sorted)
    for t in range(n):
        while ptr < m and int(ci_sorted[ptr]) == t:
            if int(ft_sorted[ptr]) > t:  # zone must survive at least this bar
                heapq.heappush(heap, (-int(ci_sorted[ptr]), ptr))
            ptr += 1

        # Drop zones that are touched at/ before t.
        while heap:
            _, p = heap[0]
            if int(ft_sorted[p]) <= t:
                heapq.heappop(heap)
                continue
            # Valid top zone
            any_active[t] = True
            last_top[t] = float(top_sorted[p])
            last_bottom[t] = float(bot_sorted[p])
            break

    return last_top, last_bottom, any_active


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


def _multi_zone_active(
    df_index: pd.DatetimeIndex,
    close: pd.Series,
    zones_df: pd.DataFrame,
    zone_type: str,
    lookback: int,
) -> pd.Series:
    """
    Multi-zone bearish active: True at bar T if close[T] is between the
    rolling-min bottom and rolling-max top of all zones of `zone_type`
    formed within the past `lookback` bars.

    Replaces single-zone _align_to_index for bearish zones where gold's
    uptrend causes the most-recent FVG to always sit above current price.
    """
    zones = zones_df[zones_df["type"] == zone_type]
    if zones.empty:
        return pd.Series(False, index=df_index)

    bot_s = pd.Series(zones["bottom"].values, index=zones["timestamp"])
    top_s = pd.Series(zones["top"].values,    index=zones["timestamp"])
    bot_s = bot_s.groupby(level=0).min().sort_index()
    top_s = top_s.groupby(level=0).max().sort_index()

    # Map each zone timestamp to the first df bar after it
    bar_idx = np.searchsorted(df_index.values, bot_s.index.values, side="right")
    valid   = bar_idx < len(df_index)

    bot_arr = np.full(len(df_index), np.nan)
    top_arr = np.full(len(df_index), np.nan)

    if valid.any():
        vi = bar_idx[valid]
        bv = bot_s.values[valid].astype(float)
        tv = top_s.values[valid].astype(float)
        tmp = pd.DataFrame({"pos": vi, "bot": bv, "top": tv})
        by_bot = tmp.groupby("pos")["bot"].min()
        by_top = tmp.groupby("pos")["top"].max()
        bot_arr[by_bot.index.values] = by_bot.values
        top_arr[by_top.index.values] = by_top.values

    bot_sparse = pd.Series(bot_arr, index=df_index)
    top_sparse = pd.Series(top_arr, index=df_index)

    roll_min_bot = bot_sparse.rolling(lookback, min_periods=1).min()
    roll_max_top = top_sparse.rolling(lookback, min_periods=1).max()

    return ((close > roll_min_bot) & (close <= roll_max_top)).fillna(False)


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
    n = len(df)

    records: list[dict] = []

    for i in bull_idx:
        bottom = float(high_arr[i - 2])
        top_   = float(low_arr[i])
        if top_ <= bottom:
            continue
        records.append({
            "type": "bullish", "top": top_, "bottom": bottom,
            "timestamp": ts[i], "status": "open",
            "timeframe": timeframe, "bpr": False, "_idx": i,
        })

    for i in bear_idx:
        top_   = float(low_arr[i - 2])
        bottom = float(high_arr[i])
        if top_ <= bottom:
            continue
        records.append({
            "type": "bearish", "top": top_, "bottom": bottom,
            "timestamp": ts[i], "status": "open",
            "timeframe": timeframe, "bpr": False, "_idx": i,
        })

    fvg_df = pd.DataFrame(records)

    # ── Time-aware lifecycle indices (offline O((n+zones) log n)) ────────────
    # We track:
    #   - first_touch_idx: first time price trades into the zone (wick touch)
    #   - invalidation_idx: first time a *close* breaks beyond the far boundary (inversion)
    #
    # For trading (per the PDFs / A+ model), a zone remains valid after first touch
    # (mitigation) until it inverts. "Active" for entries is handled in _enrich_df
    # as "price currently inside a still-valid zone".
    fvg_df["created_idx"] = fvg_df["_idx"].astype(np.int32)
    start_idx = (fvg_df["created_idx"].values + 1).astype(np.int32, copy=False)

    bull_mask2 = (fvg_df["type"].values == "bullish")
    bear_mask2 = ~bull_mask2

    first_touch = np.full(len(fvg_df), n, dtype=np.int32)
    if bull_mask2.any():
        first_touch[bull_mask2] = _first_touch_indices_threshold_leq(
            threshold=fvg_df.loc[bull_mask2, "top"].values.astype(float, copy=False),
            start_idx=start_idx[bull_mask2],
            series=low_arr.astype(float, copy=False),
        )
    if bear_mask2.any():
        first_touch[bear_mask2] = _first_touch_indices_threshold_geq(
            threshold=fvg_df.loc[bear_mask2, "bottom"].values.astype(float, copy=False),
            start_idx=start_idx[bear_mask2],
            series=high_arr.astype(float, copy=False),
        )

    fvg_df["first_touch_idx"] = first_touch

    # Invalidation (inversion) indices:
    #   bullish FVG invalidates when close < bottom
    #   bearish FVG invalidates when close > top
    inv_idx = np.full(len(fvg_df), n, dtype=np.int32)
    if bull_mask2.any():
        thr = fvg_df.loc[bull_mask2, "bottom"].values.astype(float, copy=False)
        # strict < via nextafter toward -inf
        thr = np.nextafter(thr, -np.inf)
        inv_idx[bull_mask2] = _first_touch_indices_threshold_leq(
            threshold=thr,
            start_idx=start_idx[bull_mask2],
            series=close_arr.astype(float, copy=False),
        )
    if bear_mask2.any():
        thr = fvg_df.loc[bear_mask2, "top"].values.astype(float, copy=False)
        thr = np.nextafter(thr, np.inf)
        inv_idx[bear_mask2] = _first_touch_indices_threshold_geq(
            threshold=thr,
            start_idx=start_idx[bear_mask2],
            series=close_arr.astype(float, copy=False),
        )
    fvg_df["invalidation_idx"] = inv_idx

    # Catalogue status is computed as-of end-of-dataset, but activeness is time-aware.
    status = []
    for r in fvg_df.itertuples(index=False):
        # If invalidated at any point, mark inverted. Else if ever touched, mitigated. Else open.
        if int(r.invalidation_idx) < n:
            status.append("inverted")
        elif int(r.first_touch_idx) < n:
            status.append("mitigated")
        else:
            status.append("open")
    fvg_df["status"] = status

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
                ob_type = "bullish"
                records.append({
                    "type": ob_type, "top": ob_hi, "bottom": ob_lo,
                    "timestamp": ts[ob_idx], "status": "open",
                    "timeframe": timeframe, "_idx": int(ob_idx),
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
                ob_type = "bearish"
                records.append({
                    "type": ob_type, "top": ob_hi, "bottom": ob_lo,
                    "timestamp": ts[ob_idx], "status": "open",
                    "timeframe": timeframe, "_idx": int(ob_idx),
                })

    if not records:
        return pd.DataFrame(
            columns=["type", "top", "bottom", "timestamp", "status", "timeframe"]
        )
    ob_df = pd.DataFrame(records)

    # Time-aware lifecycle indices for OBs:
    #   first_touch_idx: first retest into the OB
    #   invalidation_idx: first close breaks beyond far boundary (breaker)
    ob_df["created_idx"] = ob_df["_idx"].astype(np.int32)
    start_idx = (ob_df["created_idx"].values + 1).astype(np.int32, copy=False)
    first_touch = np.full(len(ob_df), n, dtype=np.int32)

    bull_mask2 = (ob_df["type"].values == "bullish")
    bear_mask2 = ~bull_mask2
    if bull_mask2.any():
        first_touch[bull_mask2] = _first_touch_indices_threshold_leq(
            threshold=ob_df.loc[bull_mask2, "top"].values.astype(float, copy=False),
            start_idx=start_idx[bull_mask2],
            series=low.astype(float, copy=False),
        )
    if bear_mask2.any():
        first_touch[bear_mask2] = _first_touch_indices_threshold_geq(
            threshold=ob_df.loc[bear_mask2, "bottom"].values.astype(float, copy=False),
            start_idx=start_idx[bear_mask2],
            series=high.astype(float, copy=False),
        )

    ob_df["first_touch_idx"] = first_touch

    inv_idx = np.full(len(ob_df), n, dtype=np.int32)
    if bull_mask2.any():
        thr = ob_df.loc[bull_mask2, "bottom"].values.astype(float, copy=False)
        thr = np.nextafter(thr, -np.inf)
        inv_idx[bull_mask2] = _first_touch_indices_threshold_leq(
            threshold=thr,
            start_idx=start_idx[bull_mask2],
            series=close.astype(float, copy=False),
        )
    if bear_mask2.any():
        thr = ob_df.loc[bear_mask2, "top"].values.astype(float, copy=False)
        thr = np.nextafter(thr, np.inf)
        inv_idx[bear_mask2] = _first_touch_indices_threshold_geq(
            threshold=thr,
            start_idx=start_idx[bear_mask2],
            series=close.astype(float, copy=False),
        )
    ob_df["invalidation_idx"] = inv_idx

    status: list[str] = []
    new_type: list[str] = []
    for r in ob_df.itertuples(index=False):
        tp = str(r.type)
        if int(r.invalidation_idx) < n:
            status.append("inverted")
            tp = "breaker_bearish" if tp == "bullish" else "breaker_bullish"
        elif int(r.first_touch_idx) < n:
            status.append("mitigated")
        else:
            status.append("open")
        new_type.append(tp)
    ob_df["status"] = status
    ob_df["type"] = new_type

    return ob_df.drop(columns=["_idx"])


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
        for ftype, active_col, in_label, cmp_fn, touch_any in [
            ("bullish", "fvg_bull_active", "bull",
             lambda top, bot, c: (c < top,  (c >= bot) & (c < top)),
             "low"),
            ("bearish", "fvg_bear_active", "bear",
             lambda top, bot, c: (c > bot,  (c > bot)  & (c <= top)),
             "high"),
        ]:
            zones = fvg_list[fvg_list["type"] == ftype]
            if not len(zones):
                continue

            if "created_idx" not in zones.columns:
                # Back-compat: fall back to previous sparse ffill (treat end-status 'open' as active)
                open_z = zones[zones["status"] == "open"]
                if not len(open_z):
                    continue
                top_s = _align_to_index(df.index, open_z["timestamp"], open_z["top"])
                bot_s = _align_to_index(df.index, open_z["timestamp"], open_z["bottom"])
                active_mask, in_mask = cmp_fn(top_s, bot_s, close)
                df[active_col] = active_mask.fillna(False)
                df.loc[in_mask.fillna(False), "in_fvg"] = in_label
                continue

            # Prefer invalidation_idx (zone remains valid after touch until inversion)
            expiry = zones["invalidation_idx"].values if "invalidation_idx" in zones.columns else zones["first_touch_idx"].values
            last_top, last_bot, any_active = _timeaware_active_last_zone(
                n=len(df),
                created_idx=zones["created_idx"].values,
                first_touch_idx=expiry,
                top=zones["top"].values.astype(float, copy=False),
                bottom=zones["bottom"].values.astype(float, copy=False),
            )

            top_s = pd.Series(last_top, index=df.index)
            bot_s = pd.Series(last_bot, index=df.index)
            active_mask, in_mask = cmp_fn(top_s, bot_s, close)
            # For entries, "active" means price is currently in a still-valid zone.
            in_now = in_mask.fillna(False).values
            df[active_col] = (any_active & in_now)

            # BPR exclusion applies only when currently inside a BPR zone
            if "bpr" in zones.columns and bool(zones["bpr"].any()):
                bpr_z = zones[zones["bpr"]]
                bpr_top, bpr_bot, bpr_any = _timeaware_active_last_zone(
                    n=len(df),
                    created_idx=bpr_z["created_idx"].values,
                    first_touch_idx=bpr_z["first_touch_idx"].values,
                    top=bpr_z["top"].values.astype(float, copy=False),
                    bottom=bpr_z["bottom"].values.astype(float, copy=False),
                )
                _, bpr_in = cmp_fn(pd.Series(bpr_top, index=df.index),
                                   pd.Series(bpr_bot, index=df.index), close)
                bpr_in = (bpr_any & bpr_in.fillna(False).values)
                in_mask = in_mask.fillna(False).values & ~bpr_in
                df.loc[bpr_in, "in_fvg"] = "bpr"
            else:
                in_mask = in_mask.fillna(False).values

            df.loc[in_mask, "in_fvg"] = in_label

    # ── OB columns ────────────────────────────────────────────────────────────
    if len(ob_list):
        for otype, active_col, in_label, cmp_fn in [
            ("bullish", "ob_bull_active", "bull",
             lambda hi, lo, c: (c < hi, (c >= lo) & (c < hi))),
            ("bearish", "ob_bear_active", "bear",
             lambda hi, lo, c: (c > lo, (c > lo)  & (c <= hi))),
        ]:
            zones = ob_list[ob_list["type"] == otype]
            if not len(zones):
                continue

            if "created_idx" not in zones.columns or "first_touch_idx" not in zones.columns:
                open_z = zones[zones["status"] == "open"]
                if not len(open_z):
                    continue
                top_s = _align_to_index(df.index, open_z["timestamp"], open_z["top"])
                bot_s = _align_to_index(df.index, open_z["timestamp"], open_z["bottom"])
                active_mask, in_mask = cmp_fn(top_s, bot_s, close)
                df[active_col] = active_mask.fillna(False)
                df.loc[in_mask.fillna(False), "in_ob"] = in_label
                continue

            last_top, last_bot, any_active = _timeaware_active_last_zone(
                n=len(df),
                created_idx=zones["created_idx"].values,
                first_touch_idx=zones["first_touch_idx"].values,
                top=zones["top"].values.astype(float, copy=False),
                bottom=zones["bottom"].values.astype(float, copy=False),
            )
            top_s = pd.Series(last_top, index=df.index)
            bot_s = pd.Series(last_bot, index=df.index)
            active_mask, in_mask = cmp_fn(top_s, bot_s, close)
            df[active_col] = (any_active & active_mask.fillna(False).values)
            df.loc[in_mask.fillna(False).values, "in_ob"] = in_label

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
