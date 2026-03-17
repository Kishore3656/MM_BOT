"""
backtest.py
===========
Vectorized ICT-aware backtester for signal_engine output.

Entry point
-----------
    from backtest import run_backtest, compute_stats, walk_forward, full_wf_report
    trades = run_backtest(df_5m, signals)
    stats  = compute_stats(trades)

Smoke test
----------
    python backtest.py
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd

# ── Fee / cost model ──────────────────────────────────────────────────────────
CRYPTO_FEE_PCT: float = 0.0005   # 0.05 % per side
FOREX_SPREAD:   float = 1.5      # pips (1 pip = 0.0001 for major pairs)
FOREX_PIP:      float = 0.0001

# ── Position sizing ───────────────────────────────────────────────────────────
RISK_PCT: float = 0.01           # 1 % of equity per trade

# ── Partial-close levels ──────────────────────────────────────────────────────
TP1_CLOSE_FRAC: float = 0.50     # close 50 % at tp1 (1 : 1)

# ── Walk-forward split ────────────────────────────────────────────────────────
IS_FRAC:      float = 0.70       # 70 % in-sample
OOS_THRESHOLD: float = 0.70      # flag if OOS Sharpe < 0.7 × IS Sharpe

TRADING_DAYS_PER_YEAR: int = 252


# ══════════════════════════════════════════════════════════════════════════════
#  Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _entry_cost(entry: float, size: float, instrument: str) -> float:
    """Round-trip cost at entry side only."""
    if instrument == "crypto":
        return entry * size * CRYPTO_FEE_PCT
    # forex: cost = spread × size (spread already in price units via pip)
    return FOREX_SPREAD * FOREX_PIP * size


def _exit_cost(price: float, size: float, instrument: str) -> float:
    if instrument == "crypto":
        return price * size * CRYPTO_FEE_PCT
    return FOREX_SPREAD * FOREX_PIP * size


def _bars_after(df_5m: pd.DataFrame, ts: pd.Timestamp) -> pd.DataFrame:
    """Slice df from the first bar at or after ts."""
    idx = df_5m.index.searchsorted(ts)
    return df_5m.iloc[idx:]


# ══════════════════════════════════════════════════════════════════════════════
#  Core trade simulation
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_trade(
    bars: pd.DataFrame,
    direction: str,
    entry: float,
    sl: float,
    tp1: float,
    tp2: float,
    size: float,
    instrument: str,
) -> dict:
    """
    Simulate a single trade on bar data starting from entry bar.

    Returns a dict with trade outcome metrics.
    """
    high  = bars["high"].values
    low   = bars["low"].values
    close = bars["close"].values
    times = bars.index

    cost_entry = _entry_cost(entry, size, instrument)

    remaining = 1.0   # fraction of position remaining (1.0 = 100 %)
    realized_pnl = -cost_entry
    tp1_hit  = False
    be_active = False
    be_level  = entry

    exit_time  = times[-1]
    exit_price = close[-1]
    result     = "timeout"

    for i in range(len(bars)):
        h, l, c = high[i], low[i], close[i]

        # ── TP1 check first — activates BE before SL is evaluated ────────────
        if not tp1_hit:
            if direction == "long":
                tp1_now = h >= tp1
            else:
                tp1_now = l <= tp1

            if tp1_now:
                close_size = size * TP1_CLOSE_FRAC * remaining
                if direction == "long":
                    pnl_tp1 = (tp1 - entry) * close_size
                else:
                    pnl_tp1 = (entry - tp1) * close_size
                realized_pnl += pnl_tp1 - _exit_cost(tp1, close_size, instrument)
                remaining -= TP1_CLOSE_FRAC
                tp1_hit = True
                be_active = True
                be_level  = entry

        # ── SL check — uses updated BE level if TP1 just fired this bar ──────
        current_sl = be_level if be_active else sl
        if direction == "long":
            sl_hit = l <= current_sl
        else:
            sl_hit = h >= current_sl

        # ── TP2 check ─────────────────────────────────────────────────────────
        if direction == "long":
            tp2_hit = h >= tp2
        else:
            tp2_hit = l <= tp2

        # ── SL hit ───────────────────────────────────────────────────────────
        if sl_hit:
            close_size = size * remaining
            if direction == "long":
                pnl_sl = (current_sl - entry) * close_size
            else:
                pnl_sl = (entry - current_sl) * close_size
            realized_pnl += pnl_sl - _exit_cost(current_sl, close_size, instrument)
            exit_time  = times[i]
            exit_price = current_sl
            result     = "win" if tp1_hit else "loss"
            break

        # ── TP2 hit ───────────────────────────────────────────────────────────
        if tp2_hit:
            close_size = size * remaining
            if direction == "long":
                pnl_tp2 = (tp2 - entry) * close_size
            else:
                pnl_tp2 = (entry - tp2) * close_size
            realized_pnl += pnl_tp2 - _exit_cost(tp2, close_size, instrument)
            remaining = 0.0
            exit_time  = times[i]
            exit_price = tp2
            result     = "win"
            break

    else:
        # Timeout — close at last bar
        close_size = size * remaining
        if direction == "long":
            realized_pnl += (close[-1] - entry) * close_size
        else:
            realized_pnl += (entry - close[-1]) * close_size
        realized_pnl -= _exit_cost(close[-1], close_size, instrument)

    risk = abs(entry - sl) * size
    rr   = realized_pnl / risk if risk > 1e-9 else 0.0

    return {
        "entry_time":  bars.index[0],
        "exit_time":   exit_time,
        "direction":   direction,
        "entry_price": entry,
        "stop_loss":   sl,
        "tp1":         tp1,
        "tp2":         tp2,
        "exit_price":  exit_price,
        "pnl":         realized_pnl,
        "rr":          rr,
        "result":      result,
        "tp1_hit":     tp1_hit,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public: run_backtest
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    df_5m:      pd.DataFrame,
    signals:    pd.DataFrame,
    instrument: Literal["crypto", "forex"] = "crypto",
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    """
    Simulate every signal in `signals` on `df_5m` price data.

    Parameters
    ----------
    df_5m       : enriched 5-minute OHLCV DataFrame (must cover all signal timestamps)
    signals     : DataFrame from signal_engine.generate_signals()
    instrument  : "crypto" (fee %) or "forex" (spread pips)
    initial_equity : starting account balance

    Returns
    -------
    trades      : DataFrame, one row per trade with pnl, rr, result, equity columns
    """
    required = {"signal", "entry_price", "stop_loss", "tp1", "tp2", "kill_zone"}
    missing  = required - set(signals.columns)
    if missing:
        raise ValueError(f"[backtest] signals missing columns: {missing}")

    # Kill-zone filter
    active_signals = signals[signals["kill_zone"] != "off"].copy()
    if active_signals.empty:
        warnings.warn("[backtest] No signals with active kill zone.", stacklevel=2)
        return pd.DataFrame()

    equity = initial_equity
    trades = []

    for row in active_signals.itertuples():
        direction  = "long" if row.signal == 1 else "short"
        entry      = row.entry_price
        sl         = row.stop_loss
        tp1        = row.tp1
        tp2        = row.tp2
        bar_time   = row.Index  # DatetimeIndex

        if pd.isna(entry) or pd.isna(sl) or pd.isna(tp1) or pd.isna(tp2):
            continue

        # Position size: risk 1 % of current equity
        risk_amount = equity * RISK_PCT
        distance    = abs(entry - sl)
        if distance < 1e-9:
            continue
        size = risk_amount / distance

        future_bars = _bars_after(df_5m, bar_time)
        if len(future_bars) < 2:
            continue

        trade = _simulate_trade(
            future_bars, direction, entry, sl, tp1, tp2, size, instrument
        )
        trade["equity_before"] = equity
        equity += trade["pnl"]
        trade["equity_after"]  = equity
        trade["kill_zone"] = row.kill_zone

        # carry through extra signal columns if available
        for col in ("rr_ratio", "confluence", "c1", "c2", "c3", "c4", "c5", "c6", "c7"):
            if hasattr(row, col):
                trade[col] = getattr(row, col)

        trades.append(trade)

    if not trades:
        return pd.DataFrame()

    trades_df = pd.DataFrame(trades)
    trades_df = trades_df.set_index("entry_time")
    return trades_df


# ══════════════════════════════════════════════════════════════════════════════
#  Public: compute_stats
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(trades: pd.DataFrame) -> dict:
    """
    Compute performance statistics from a trades DataFrame.

    Returns
    -------
    dict with keys:
        total_trades, win_rate, profit_factor, avg_rr_win, avg_rr_loss,
        sharpe, max_drawdown_pct, total_return_pct,
        by_kill_zone (dict), monthly_returns (Series)
    """
    if trades.empty:
        return {"error": "no trades"}

    n        = len(trades)
    wins     = trades[trades["result"] == "win"]
    losses   = trades[trades["result"] == "loss"]
    win_rate = len(wins) / n

    gross_win  = wins["pnl"].sum()
    gross_loss = abs(losses["pnl"].sum())
    pf         = gross_win / gross_loss if gross_loss > 1e-9 else np.inf

    avg_rr_win  = wins["rr"].mean()  if len(wins)   else 0.0
    avg_rr_loss = losses["rr"].mean() if len(losses) else 0.0

    # Equity curve for Sharpe / drawdown
    eq = trades["equity_after"].values
    eq_series = pd.Series(eq, index=trades.index)

    # Daily returns from equity curve
    daily_eq = eq_series.resample("1D").last().ffill().dropna()
    daily_ret = daily_eq.pct_change().dropna()
    sharpe = (
        (daily_ret.mean() / daily_ret.std()) * np.sqrt(TRADING_DAYS_PER_YEAR)
        if daily_ret.std() > 1e-9
        else 0.0
    )

    # Max drawdown
    rolling_max = eq_series.cummax()
    drawdown    = (eq_series - rolling_max) / rolling_max
    max_dd      = float(drawdown.min()) * 100.0   # negative %

    # Total return
    total_return_pct = (eq[-1] - trades["equity_before"].iloc[0]) / trades["equity_before"].iloc[0] * 100

    # By kill zone
    by_kz = {}
    if "kill_zone" in trades.columns:
        for kz, grp in trades.groupby("kill_zone"):
            grp_wins = (grp["result"] == "win").sum()
            by_kz[kz] = {
                "trades":   len(grp),
                "win_rate": grp_wins / len(grp),
                "net_pnl":  grp["pnl"].sum(),
            }

    # Monthly returns
    monthly_eq  = eq_series.resample("ME").last().ffill()
    monthly_ret = monthly_eq.pct_change().dropna() * 100.0

    return {
        "total_trades":     n,
        "win_rate":         round(win_rate, 4),
        "profit_factor":    round(pf, 4),
        "avg_rr_win":       round(float(avg_rr_win),  4),
        "avg_rr_loss":      round(float(avg_rr_loss), 4),
        "sharpe":           round(float(sharpe),         4),
        "max_drawdown_pct": round(max_dd,               4),
        "total_return_pct": round(float(total_return_pct), 4),
        "by_kill_zone":     by_kz,
        "monthly_returns":  monthly_ret,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public: walk_forward
# ══════════════════════════════════════════════════════════════════════════════

def walk_forward(
    df_5m:    pd.DataFrame,
    signals:  pd.DataFrame,
    instrument: Literal["crypto", "forex"] = "crypto",
    initial_equity: float = 10_000.0,
) -> dict:
    """
    70/30 in-sample / out-of-sample split.

    Returns
    -------
    dict with keys: is_stats, oos_stats, overfit_flag, split_date
    """
    if signals.empty:
        return {"error": "no signals"}

    sig_sorted = signals.sort_index()
    n_total    = len(sig_sorted)
    split_idx  = int(n_total * IS_FRAC)
    split_date = sig_sorted.index[split_idx]

    is_signals  = sig_sorted.iloc[:split_idx]
    oos_signals = sig_sorted.iloc[split_idx:]

    is_trades  = run_backtest(df_5m, is_signals,  instrument, initial_equity)
    oos_trades = run_backtest(df_5m, oos_signals, instrument, initial_equity)

    is_stats  = compute_stats(is_trades)
    oos_stats = compute_stats(oos_trades)

    is_sharpe  = is_stats.get("sharpe",  0.0)
    oos_sharpe = oos_stats.get("sharpe", 0.0)

    overfit = bool(
        isinstance(is_sharpe, (int, float)) and
        isinstance(oos_sharpe, (int, float)) and
        is_sharpe > 0 and
        oos_sharpe < OOS_THRESHOLD * is_sharpe
    )

    return {
        "split_date":  split_date,
        "is_stats":    is_stats,
        "oos_stats":   oos_stats,
        "overfit_flag": overfit,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Public: full_wf_report
# ══════════════════════════════════════════════════════════════════════════════

def full_wf_report(
    df_5m:    pd.DataFrame,
    signals:  pd.DataFrame,
    instrument: Literal["crypto", "forex"] = "crypto",
    initial_equity: float = 10_000.0,
    plot: bool = True,
) -> dict:
    """
    Run full backtest + walk-forward and print a summary report.
    Optionally plot equity curve + monthly returns heatmap.

    Returns aggregated stats dict.
    """
    print("=" * 60)
    print("  ICT Backtest Report")
    print("=" * 60)

    all_trades = run_backtest(df_5m, signals, instrument, initial_equity)
    stats      = compute_stats(all_trades)

    if "error" in stats:
        print("  No trades generated.")
        return stats

    print(f"\n  Total trades    : {stats['total_trades']}")
    print(f"  Win rate        : {stats['win_rate']*100:.1f}%")
    print(f"  Profit factor   : {stats['profit_factor']:.2f}")
    print(f"  Avg RR (wins)   : {stats['avg_rr_win']:.2f}")
    print(f"  Avg RR (losses) : {stats['avg_rr_loss']:.2f}")
    print(f"  Sharpe (ann.)   : {stats['sharpe']:.2f}")
    print(f"  Max drawdown    : {stats['max_drawdown_pct']:.2f}%")
    print(f"  Total return    : {stats['total_return_pct']:.2f}%")

    if stats.get("by_kill_zone"):
        print("\n  By Kill Zone:")
        for kz, kz_stats in stats["by_kill_zone"].items():
            print(f"    {kz:10s}  trades={kz_stats['trades']:4d}  "
                  f"wr={kz_stats['win_rate']*100:.1f}%  "
                  f"pnl={kz_stats['net_pnl']:+.2f}")

    # Walk-forward
    wf = walk_forward(df_5m, signals, instrument, initial_equity)
    if "error" not in wf:
        print(f"\n  Walk-forward (split {wf['split_date'].date()})")
        print(f"    IS  Sharpe={wf['is_stats']['sharpe']:.2f}  "
              f"return={wf['is_stats']['total_return_pct']:.1f}%")
        print(f"    OOS Sharpe={wf['oos_stats']['sharpe']:.2f}  "
              f"return={wf['oos_stats']['total_return_pct']:.1f}%")
        if wf["overfit_flag"]:
            print("  *** WARNING: likely overfit — OOS Sharpe < 70% of IS Sharpe ***")

    if plot:
        _plot_results(all_trades, stats)

    print("=" * 60)
    return {"full": stats, "walk_forward": wf, "trades": all_trades}


# ══════════════════════════════════════════════════════════════════════════════
#  Plotting helpers
# ══════════════════════════════════════════════════════════════════════════════

def _plot_results(trades: pd.DataFrame, stats: dict) -> None:
    try:
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
    except ImportError:
        warnings.warn("[backtest] matplotlib not installed — skipping plots.", stacklevel=2)
        return

    fig, axes = plt.subplots(2, 1, figsize=(14, 9))
    fig.suptitle("ICT Bot — Backtest Results", fontsize=14)

    # ── Equity curve ─────────────────────────────────────────────────────────
    ax1 = axes[0]
    eq_curve = trades["equity_after"]
    ax1.plot(eq_curve.index, eq_curve.values, linewidth=1.5, color="#2196F3")
    ax1.fill_between(eq_curve.index, eq_curve.values,
                     eq_curve.values[0], alpha=0.15, color="#2196F3")
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Equity ($)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.grid(True, alpha=0.3)

    # ── Monthly returns heatmap ───────────────────────────────────────────────
    monthly = stats.get("monthly_returns")
    ax2 = axes[1]

    if monthly is not None and not monthly.empty:
        monthly_df = monthly.copy()
        monthly_df.index = pd.to_datetime(monthly_df.index)
        table = monthly_df.groupby([
            monthly_df.index.year,
            monthly_df.index.month
        ]).sum().unstack(level=1)
        table.columns = [
            "Jan","Feb","Mar","Apr","May","Jun",
            "Jul","Aug","Sep","Oct","Nov","Dec"
        ][:table.shape[1]]

        vmax = max(abs(table.values[~np.isnan(table.values)]).max(), 0.01)
        im = ax2.imshow(table.values, aspect="auto", cmap="RdYlGn",
                        vmin=-vmax, vmax=vmax)
        ax2.set_xticks(range(len(table.columns)))
        ax2.set_xticklabels(table.columns, fontsize=9)
        ax2.set_yticks(range(len(table.index)))
        ax2.set_yticklabels(table.index, fontsize=9)
        ax2.set_title("Monthly Returns (%)")
        for y in range(table.shape[0]):
            for x in range(table.shape[1]):
                val = table.values[y, x]
                if not np.isnan(val):
                    ax2.text(x, y, f"{val:.1f}", ha="center", va="center",
                             fontsize=7, color="black")
        plt.colorbar(im, ax=ax2, fraction=0.02, label="%")
    else:
        ax2.text(0.5, 0.5, "No monthly return data", transform=ax2.transAxes,
                 ha="center", va="center")
        ax2.set_title("Monthly Returns (%)")

    plt.tight_layout()
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  Smoke test  (python backtest.py)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path

    print("=== backtest.py smoke test ===\n")

    script_dir = Path(__file__).parent
    local_path = script_dir / "gold_clean_data"

    if not local_path.exists():
        print(f"Skipped — '{local_path}' not found.")
        sys.exit(0)

    try:
        from data_feed import fetch_mtf
        from market_structure import add_market_structure
        from dealing_range import compute_dealing_range
        from liquidity import add_liquidity
        from pd_arrays import add_pd_arrays
        from signal_engine import generate_signals
    except ImportError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print("Loading XAU data …")
    mtf = fetch_mtf(
        "XAU/USD",
        source="local",
        local_dir=str(local_path),
        local_prefix="XAU",
    )

    enriched = {}
    for tf, df in mtf.items():
        print(f"  Enriching [{tf}] …")
        df = add_market_structure(df, pivot_n=5)
        df = compute_dealing_range(df, window="session")
        df = add_liquidity(df, lookback=20)
        df, _, _ = add_pd_arrays(df, timeframe=tf)
        enriched[tf] = df

    # Debug: HTF structure distribution
    df4h = enriched.get("4h")
    if df4h is not None:
        counts = df4h["structure"].value_counts()
        print(f"\n  4H structure distribution:\n{counts.to_string()}")

    # Debug: short condition hit rates on bearish-HTF bars
    from signal_engine import _build_conditions, _align_htf
    import numpy as _np
    df5m = enriched["5m"]
    _htf_struct = _align_htf(enriched["4h"], df5m.index, ["structure"])["structure"]
    _conds = _build_conditions(df5m, _htf_struct, choch_lookback=10)
    _bear_mask = _htf_struct == "bearish"
    print("\n  Short condition hit rates (on bearish HTF bars):")
    for _name, (_cl, _cs) in _conds.items():
        rate = _cs[_bear_mask].mean()
        print(f"    {_name}: {rate:.1%}")

    print("\nGenerating signals …")
    signals = generate_signals(enriched)
    print(f"  Total A+ signals: {len(signals)}")

    if signals.empty:
        print("  No signals — cannot run backtest.")
        sys.exit(0)

    direction_counts = signals["signal"].value_counts().to_dict()
    print(f"  Direction split : {direction_counts}")

    result = full_wf_report(
        enriched["5m"],
        signals,
        instrument="crypto",
        initial_equity=10_000.0,
        plot=False,
    )

    trades = result.get("trades")
    if trades is not None and not trades.empty:
        print("\n  Last 5 trades:")
        cols = ["direction", "entry_price", "exit_price", "pnl", "rr", "result"]
        available = [c for c in cols if c in trades.columns]
        print(trades[available].tail().to_string())
