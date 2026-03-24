"""
param_sweep.py
==============
Grid-search key signal_engine parameters across a basket of assets.
Tests each parameter in isolation (one factor at a time) while holding others at baseline.

Usage
-----
    python param_sweep.py
    python param_sweep.py --assets EURUSD=X GBPUSD=X BTC/USDT
    python param_sweep.py --full        # all combinations (slower)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import warnings
import itertools
from contextlib import contextmanager, redirect_stdout, redirect_stderr
from copy import deepcopy

warnings.filterwarnings("ignore")

import signal_engine as _se

# ── Baseline (current defaults) ───────────────────────────────────────────────
BASELINE = {
    "ADX_THRESHOLD":         20.0,
    "MIN_RR":                2.5,
    "MIN_CONDITIONS":        6,
    "CHOCH_LOOKBACK":        10,
    "SWEEP_LOOKBACK":        24,
    "DISPLACEMENT_ATR_MULT": 1.2,
    "SL_ATR_BUFFER":         0.50,
}

# ── Parameter grid (one-at-a-time sweep) ─────────────────────────────────────
PARAM_GRID = {
    "ADX_THRESHOLD":         [15.0, 20.0, 25.0, 30.0],
    "MIN_RR":                [2.0, 2.5, 3.0],
    "MIN_CONDITIONS":        [5, 6, 7],
    "CHOCH_LOOKBACK":        [5, 10, 15, 20],
    "SWEEP_LOOKBACK":        [12, 24, 36],
    "DISPLACEMENT_ATR_MULT": [1.0, 1.2, 1.5, 2.0],
    "SL_ATR_BUFFER":         [0.25, 0.50, 0.75, 1.0],
}

DEFAULT_ASSETS = ["EURUSD=X", "GBPUSD=X", "BTC/USDT", "GC=F"]


def _apply_params(params: dict) -> None:
    """Write parameter values into the signal_engine module globals."""
    for k, v in params.items():
        setattr(_se, k, v)


@contextmanager
def _silence():
    """Suppress all print/stdout output during backtest runs."""
    with open(os.devnull, "w", encoding="utf-8", errors="replace") as null:
        with redirect_stdout(null), redirect_stderr(null):
            yield


def _run_combo(params: dict, assets: list[str]) -> dict:
    """Apply params, run backtest on all assets, return aggregated stats."""
    _apply_params(params)

    from run_backtest_multi import run_asset, ASSET_REGISTRY

    all_trades = 0
    all_wins   = 0
    pf_sum     = 0.0
    pf_count   = 0
    sharpe_sum = 0.0
    ret_sum    = 0.0
    dd_sum     = 0.0
    valid      = 0

    for sym in assets:
        meta = ASSET_REGISTRY.get(sym, {})
        instrument = meta.get("instrument", "crypto")
        try:
            with _silence():
                result = run_asset(sym, instrument, initial_equity=10_000, plot=False)
        except Exception as e:
            print(f"    [{sym}] ERROR: {e}")
            continue

        if "error" in result or not result.get("full"):
            continue

        stats = result["full"]
        t = stats.get("total_trades", 0)
        if t == 0:
            continue

        all_trades += t
        all_wins   += int(stats.get("win_rate", 0) * t)
        pf = stats.get("profit_factor", 0)
        if pf < 99:   # skip inf
            pf_sum   += pf
            pf_count += 1
        sharpe_sum += stats.get("sharpe", 0)
        ret_sum    += stats.get("total_return_pct", 0)
        dd_sum     += abs(stats.get("max_drawdown_pct", 0))
        valid      += 1

    if all_trades == 0:
        return {"trades": 0, "win_rate": 0, "pf": 0, "sharpe": 0, "ret": 0, "dd": 0}

    return {
        "trades":   all_trades,
        "win_rate": all_wins / all_trades if all_trades else 0,
        "pf":       pf_sum / pf_count if pf_count else float("inf"),
        "sharpe":   sharpe_sum / valid if valid else 0,
        "ret":      ret_sum / valid if valid else 0,
        "dd":       dd_sum / valid if valid else 0,
    }


def _score(r: dict) -> float:
    """Composite score: prioritise win rate × PF, penalise drawdown."""
    if r["trades"] < 3:
        return -999.0
    wr  = r["win_rate"]
    pf  = min(r["pf"], 5.0)          # cap at 5× to avoid inf dominating
    sh  = max(r["sharpe"], -3.0)
    dd  = r["dd"]
    return (wr * 0.35 + pf * 0.25 + sh * 0.20) - dd * 0.20


def sweep_one_at_a_time(assets: list[str]) -> list[dict]:
    """Test each parameter independently; hold all others at baseline."""
    rows = []
    for param, values in PARAM_GRID.items():
        print(f"\n{'='*60}\n  Sweeping: {param}\n{'='*60}")
        for val in values:
            params = {**BASELINE, param: val}
            tag = "★ baseline" if val == BASELINE[param] else ""
            print(f"  {param}={val} {tag}")
            result = _run_combo(params, assets)
            rows.append({
                "param":     param,
                "value":     val,
                "baseline":  val == BASELINE[param],
                "trades":    result["trades"],
                "win_rate":  result["win_rate"],
                "pf":        result["pf"],
                "sharpe":    result["sharpe"],
                "ret_pct":   result["ret"],
                "max_dd":    result["dd"],
                "score":     _score(result),
            })
    return rows


def sweep_full_grid(assets: list[str], keys: list[str]) -> list[dict]:
    """Exhaustive grid — exponential, use only with a few params."""
    rows = []
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    total  = len(combos)
    print(f"\nFull grid: {total} combinations over {keys}")
    for i, vals in enumerate(combos, 1):
        params = {**BASELINE, **dict(zip(keys, vals))}
        print(f"  [{i:3d}/{total}]  " + "  ".join(f"{k}={v}" for k, v in zip(keys, vals)))
        result = _run_combo(params, assets)
        rows.append({
            **{k: v for k, v in zip(keys, vals)},
            "trades":   result["trades"],
            "win_rate": result["win_rate"],
            "pf":       result["pf"],
            "sharpe":   result["sharpe"],
            "ret_pct":  result["ret"],
            "max_dd":   result["dd"],
            "score":    _score(result),
        })
    return rows


def _print_table(rows: list[dict], title: str) -> None:
    if not rows:
        print("No results.")
        return
    import pandas as pd
    df = pd.DataFrame(rows).sort_values("score", ascending=False)

    # Format
    df["win_rate"] = (df["win_rate"] * 100).round(1).astype(str) + "%"
    df["pf"]       = df["pf"].apply(lambda x: f"{x:.2f}" if x < 99 else "inf")
    df["sharpe"]   = df["sharpe"].round(2)
    df["ret_pct"]  = df["ret_pct"].round(2).astype(str) + "%"
    df["max_dd"]   = df["max_dd"].round(2).astype(str) + "%"
    df["score"]    = df["score"].round(3)
    if "baseline" in df.columns:
        df["baseline"] = df["baseline"].map({True: "★", False: ""})

    print(f"\n{'='*100}")
    print(f"  {title}")
    print("="*100)
    print(df.to_string(index=False))
    print()

    # Best per parameter
    if "param" in df.columns:
        print("  ── Best value per parameter ──")
        for param in df["param"].unique():
            sub = df[df["param"] == param].copy()
            # Re-sort by numeric score (string formatted above, need original)
            best = rows  # use original rows for numeric comparison
            param_rows = [r for r in best if r.get("param") == param]
            best_row = max(param_rows, key=lambda r: r["score"])
            print(f"    {param:25s}  best={best_row['value']}  "
                  f"score={best_row['score']:.3f}  "
                  f"wr={best_row['win_rate']*100:.1f}%  "
                  f"pf={best_row['pf']:.2f}  "
                  f"trades={best_row['trades']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", nargs="+", default=DEFAULT_ASSETS)
    parser.add_argument("--full", action="store_true",
                        help="Full grid on ADX_THRESHOLD × MIN_RR × MIN_CONDITIONS")
    args = parser.parse_args()

    print(f"\nParameter sweep on assets: {args.assets}")
    print(f"Baseline: {BASELINE}\n")

    if args.full:
        keys  = ["ADX_THRESHOLD", "MIN_RR", "MIN_CONDITIONS"]
        rows  = sweep_full_grid(args.assets, keys)
        _print_table(rows, "FULL GRID — ADX × MIN_RR × MIN_CONDITIONS")
    else:
        rows = sweep_one_at_a_time(args.assets)
        _print_table(rows, "ONE-AT-A-TIME PARAMETER SWEEP")

    # Restore baseline
    _apply_params(BASELINE)
    print("\nBaseline parameters restored.")


if __name__ == "__main__":
    main()
