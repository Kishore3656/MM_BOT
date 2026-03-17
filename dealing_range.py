"""
dealing_range.py
================
ICT dealing range and optimal trade entry (OTE) helpers.

Adds to each OHLCV DataFrame:
  range_high   - causal dealing range high
  range_low    - causal dealing range low
  equilibrium  - midpoint of the dealing range
  premium      - True when close is above equilibrium
  discount     - True when close is below equilibrium
  ote_low      - lower bound of the OTE band for long setups
  ote_high     - upper bound of the OTE band for long setups
  ote_short_low  - lower bound of the OTE band for short setups
  ote_short_high - upper bound of the OTE band for short setups

Entry point
-----------
    from dealing_range import compute_dealing_range, in_ote_zone, dealing_zone
    df = compute_dealing_range(df, window="session")
"""

from __future__ import annotations

from typing import Literal

import pandas as pd

SESSION_WINDOWS: list[tuple[str, float, float]] = [
    ("london", 7.0, 10.0),
    ("newyork", 13.0, 16.0),
    ("nypm", 18.5, 21.0),
]

LONG_OTE_MIN = 0.62
LONG_OTE_MAX = 0.79


def _validate_ohlc(df: pd.DataFrame) -> None:
    required = {"high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"[dealing_range] Missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("[dealing_range] DataFrame index must be a DatetimeIndex.")


def _session_labels(index: pd.DatetimeIndex) -> pd.Series:
    hour = index.hour + index.minute / 60.0
    labels = pd.Series("off", index=index, dtype="object")

    for name, start, end in SESSION_WINDOWS:
        mask = (hour >= start) & (hour < end)
        labels.loc[mask] = name

    return labels


def _session_groups(index: pd.DatetimeIndex) -> pd.Series:
    labels = _session_labels(index)
    return labels.ne(labels.shift(1)).cumsum()


def _causal_session_range(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    groups = _session_groups(df.index)
    range_high = df["high"].groupby(groups).cummax()
    range_low = df["low"].groupby(groups).cummin()
    return range_high, range_low


def _causal_rolling_range(df: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
    if window <= 0:
        raise ValueError("[dealing_range] Numeric window must be greater than 0.")
    range_high = df["high"].rolling(window=window, min_periods=1).max()
    range_low = df["low"].rolling(window=window, min_periods=1).min()
    return range_high, range_low


def _ote_bounds(range_high: pd.Series, range_low: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    span = range_high - range_low

    long_ote_low = range_high - (span * LONG_OTE_MAX)
    long_ote_high = range_high - (span * LONG_OTE_MIN)

    short_ote_low = range_low + (span * LONG_OTE_MIN)
    short_ote_high = range_low + (span * LONG_OTE_MAX)

    return long_ote_low, long_ote_high, short_ote_low, short_ote_high


def compute_dealing_range(df: pd.DataFrame, window: str | int = "session") -> pd.DataFrame:
    """
    Add causal ICT dealing range and OTE columns.

    Parameters
    ----------
    df : DataFrame
        Must contain high, low, close columns and a DatetimeIndex.
    window : "session" or int
        "session" computes the active range within the current contiguous session.
        An integer computes a trailing range over the last N bars.
    """
    _validate_ohlc(df)

    out = df.copy()

    if window == "session":
        range_high, range_low = _causal_session_range(out)
    elif isinstance(window, int):
        range_high, range_low = _causal_rolling_range(out, window)
    else:
        raise ValueError("[dealing_range] window must be 'session' or a positive integer.")

    equilibrium = (range_high + range_low) / 2.0
    premium = out["close"] > equilibrium
    discount = out["close"] < equilibrium
    ote_low, ote_high, ote_short_low, ote_short_high = _ote_bounds(range_high, range_low)

    out["range_high"] = range_high
    out["range_low"] = range_low
    out["equilibrium"] = equilibrium
    out["premium"] = premium.fillna(False)
    out["discount"] = discount.fillna(False)
    out["ote_low"] = ote_low
    out["ote_high"] = ote_high
    out["ote_short_low"] = ote_short_low
    out["ote_short_high"] = ote_short_high

    return out


def in_ote_zone(
    price: float,
    direction: Literal["long", "short"],
    range_high: float,
    range_low: float,
) -> bool:
    """Return True when price is inside the ICT OTE band for the given direction."""
    if pd.isna(price) or pd.isna(range_high) or pd.isna(range_low):
        return False
    if range_high < range_low:
        raise ValueError("[dealing_range] range_high must be greater than or equal to range_low.")

    span = range_high - range_low
    if direction == "long":
        zone_low = range_high - (span * LONG_OTE_MAX)
        zone_high = range_high - (span * LONG_OTE_MIN)
    elif direction == "short":
        zone_low = range_low + (span * LONG_OTE_MIN)
        zone_high = range_low + (span * LONG_OTE_MAX)
    else:
        raise ValueError("[dealing_range] direction must be 'long' or 'short'.")

    return bool(zone_low <= price <= zone_high)


def dealing_zone(price: float, range_high: float, range_low: float) -> Literal["premium", "discount", "equilibrium"]:
    """Classify price relative to the dealing range midpoint."""
    if pd.isna(price) or pd.isna(range_high) or pd.isna(range_low):
        return "equilibrium"
    if range_high < range_low:
        raise ValueError("[dealing_range] range_high must be greater than or equal to range_low.")

    midpoint = (range_high + range_low) / 2.0
    if price > midpoint:
        return "premium"
    if price < midpoint:
        return "discount"
    return "equilibrium"


if __name__ == "__main__":
    idx = pd.date_range("2026-01-05 06:00:00", periods=8, freq="h", tz="UTC")
    sample = pd.DataFrame(
        {
            "high": [100, 102, 103, 105, 104, 107, 106, 108],
            "low": [99, 100, 101, 102, 101, 103, 102, 104],
            "close": [99.5, 101, 102, 103, 102.5, 106, 104, 107],
        },
        index=idx,
    )

    enriched = compute_dealing_range(sample, window="session")
    print(enriched[["range_high", "range_low", "equilibrium", "ote_low", "ote_high"]].tail())
