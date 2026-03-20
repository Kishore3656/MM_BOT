"""
bot.py
======
Live trading loop that ties all modules together.

Usage
-----
  # Paper trade BTC/USDT (crypto, ccxt/Binance, 5m exec)
  python bot.py --symbol BTC/USDT --paper

  # Paper trade Gold futures (yfinance, 1h exec)
  python bot.py --symbol GC=F --source yfinance --paper

  # Paper trade EURUSD (yfinance, 1h exec)
  python bot.py --symbol EURUSD=X --source yfinance --paper

  # Live ccxt with Telegram alerts
  python bot.py --symbol BTC/USDT --source ccxt --telegram

  # Run one cycle only (no scheduler)
  python bot.py --symbol BTC/USDT --paper --run-once

Notes
-----
- Paper mode simulates orders/positions without broker connectivity.
- yfinance source is paper-only (yfinance has no order API).
- Execution timeframe auto-detection:
    ccxt sources   → 5m exec  (Binance has deep 5m history)
    yfinance sources → 1h exec (yfinance 5m is capped at 60 days)
- The pipeline exactly mirrors run_backtest_multi.py's run_asset() to ensure
  live signals match backtest signals.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.cron import CronTrigger
    _HAS_APSCHEDULER = True
except Exception:
    BackgroundScheduler = None  # type: ignore
    CronTrigger = None          # type: ignore
    _HAS_APSCHEDULER = False

from data_feed import fetch_mtf, _add_swings, _add_session_column
from market_structure import add_market_structure
from dealing_range import compute_dealing_range
from liquidity import add_liquidity, detect_sweeps, detect_judas_swing
from pd_arrays import add_pd_arrays
from signal_engine import get_signal, MIN_CONDITIONS
from risk_manager import ICTRiskManager

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

try:
    from telegram import Bot as _TGBot  # type: ignore
    _HAS_TELEGRAM = True
except Exception:
    _TGBot = None  # type: ignore
    _HAS_TELEGRAM = False

try:
    import ccxt  # type: ignore
except Exception:
    ccxt = None  # type: ignore

try:
    from oandapyV20 import API as _OandaAPI  # type: ignore
except Exception:
    _OandaAPI = None  # type: ignore

try:
    import yfinance as yf  # type: ignore
    _HAS_YFINANCE = True
except Exception:
    yf = None  # type: ignore
    _HAS_YFINANCE = False


# ── Swing-N per timeframe (matches data_feed constants) ──────────────────────
_SWING_N = {"4h": 10, "1h": 5, "5m": 5}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _setup_logger(log_path: str = "bot.log") -> logging.Logger:
    logger = logging.getLogger("ict_bot")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)sZ %(levelname)s %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _floor_to_5m(ts: datetime) -> datetime:
    ts = ts.astimezone(timezone.utc)
    minute = (ts.minute // 5) * 5
    return ts.replace(minute=minute, second=0, microsecond=0)


def _next_5m_close(ts: datetime) -> datetime:
    base = _floor_to_5m(ts)
    return base + timedelta(minutes=5)


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _normalise_yf_df(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise a raw yfinance DataFrame to pipeline format."""
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns = [c.strip().lower() for c in df.columns]
    df.rename(columns={"adj close": "close", "adj_close": "close"}, inplace=True)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"yfinance DataFrame missing columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
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


def _fetch_yfinance_mtf(symbol: str) -> dict[str, pd.DataFrame]:
    """
    Fetch fresh 4h/1h data for a yfinance asset.
    Returns {"4h": df, "1h": df} — no "5m" key (1h is the exec frame).
    """
    if not _HAS_YFINANCE:
        raise ImportError("yfinance is not installed. Run: pip install yfinance")

    df1h = yf.download(symbol, period="2y", interval="1h",
                       auto_adjust=True, progress=False, threads=False)
    if df1h.empty:
        raise RuntimeError(f"[yfinance] No 1h data for {symbol}")
    df1h = _normalise_yf_df(df1h)

    df4h = df1h.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"}
    ).dropna(subset=["open"])

    # Add swing highs/lows and session label (mirrors data_feed.fetch_mtf)
    for tf, df in [("4h", df4h), ("1h", df1h)]:
        df = _add_swings(df, _SWING_N[tf])
        df = _add_session_column(df)
        if tf == "4h":
            df4h = df
        else:
            df1h = df

    return {"4h": df4h, "1h": df1h}


def _forward_fill_htf_onto_exec(
    df_htf: pd.DataFrame, df_exec: pd.DataFrame
) -> pd.DataFrame:
    """
    Forward-fill HTF zone/level columns onto the execution frame, then
    re-run sweep detection and PD arrays on the exec frame.
    Mirrors run_backtest_multi._forward_fill_htf_onto_exec exactly.
    """
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
    df_exec = pd.concat([df_exec.drop(columns=overlap), aligned], axis=1)

    df_exec = detect_sweeps(df_exec)
    df_exec = detect_judas_swing(df_exec)
    df_exec, _, _ = add_pd_arrays(df_exec, timeframe="5m")
    return df_exec


# ══════════════════════════════════════════════════════════════════════════════
#  Telegram notifier
# ══════════════════════════════════════════════════════════════════════════════

class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, logger: logging.Logger):
        if not _HAS_TELEGRAM:
            raise ImportError("python-telegram-bot is not installed.")
        self.bot = _TGBot(token=token)
        self.chat_id = chat_id
        self.logger = logger

    def send(self, text: str) -> None:
        try:
            self.bot.send_message(chat_id=self.chat_id, text=text)
        except Exception as e:
            self.logger.warning(f"[telegram] send failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Order / Position dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Position:
    symbol: str
    side: str        # "long" | "short"
    units: float
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    opened_at: datetime
    status: str = "open"   # open | closed
    tp1_hit: bool = False
    stop_at_be: bool = False


@dataclass
class PendingOrder:
    symbol: str
    side: str        # "long" | "short"
    units: float
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    created_at: datetime
    status: str = "pending"  # pending | filled | cancelled


# ══════════════════════════════════════════════════════════════════════════════
#  Broker abstraction
# ══════════════════════════════════════════════════════════════════════════════

class BrokerBase:
    def get_equity(self) -> float:
        raise NotImplementedError

    def get_data_client(self):
        """Return exchange/API client for data_feed.fetch_mtf(), or None."""
        return None

    def place_bracket_order(self, pos: Position) -> dict[str, Any]:
        raise NotImplementedError

    def cancel_all(self) -> None:
        return None

    def get_last_price(self, symbol: str) -> float | None:
        return None


class PaperBroker(BrokerBase):
    def __init__(self, initial_equity: float, logger: logging.Logger):
        self.equity = float(initial_equity)
        self.logger = logger
        self.last_price: dict[str, float] = {}

    def get_equity(self) -> float:
        return float(self.equity)

    def set_last_price(self, symbol: str, price: float) -> None:
        self.last_price[str(symbol)] = float(price)

    def get_last_price(self, symbol: str) -> float | None:
        return self.last_price.get(str(symbol))

    def place_bracket_order(self, pos: Position) -> dict[str, Any]:
        self.logger.info(
            f"[PAPER] Open {pos.side} {pos.symbol} units={pos.units:.6f} "
            f"entry={pos.entry:.5f} sl={pos.stop_loss:.5f} "
            f"tp1={pos.tp1:.5f} tp2={pos.tp2:.5f}"
        )
        return {"mode": "paper", "status": "placed"}


class CCXTBroker(BrokerBase):
    def __init__(self, exchange_id: str, logger: logging.Logger, *,
                 allow_unsafe_brackets: bool = False):
        if ccxt is None:
            raise ImportError("ccxt is required for CCXTBroker.")
        self.logger = logger
        self.allow_unsafe_brackets = bool(allow_unsafe_brackets)
        ex_cls = getattr(ccxt, exchange_id)
        self.ex = ex_cls({"enableRateLimit": True})
        key = os.getenv("CCXT_API_KEY")
        secret = os.getenv("CCXT_API_SECRET")
        if key and secret:
            self.ex.apiKey = key
            self.ex.secret = secret

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            if "USDT" in bal.get("total", {}):
                return float(bal["total"]["USDT"])
            total = bal.get("total") or {}
            if total:
                vals = [v for v in total.values() if isinstance(v, (int, float))]
                return float(max(vals)) if vals else 0.0
        except Exception as e:
            self.logger.error(f"[ccxt] fetch_balance failed: {e}")
        return 0.0

    def get_data_client(self):
        return self.ex

    def get_last_price(self, symbol: str) -> float | None:
        try:
            t = self.ex.fetch_ticker(symbol)
            return float(t.get("last")) if t and t.get("last") is not None else None
        except Exception as e:
            self.logger.error(f"[ccxt] fetch_ticker failed: {e}")
            return None

    def place_bracket_order(self, pos: Position) -> dict[str, Any]:
        """
        Best-effort bracket via ccxt.
        Guard by --allow-unsafe-live because stop/TP order types are exchange-specific.
        """
        if not self.allow_unsafe_brackets:
            return {
                "status": "blocked",
                "error": (
                    "ccxt bracket orders are exchange-specific. "
                    "Run with --allow-unsafe-live to attempt generic stop/take-profit orders."
                ),
            }
        side = "buy" if pos.side == "long" else "sell"
        opp  = "sell" if side == "buy" else "buy"
        out: dict[str, Any] = {"entry": None, "sl": None, "tp": None}
        try:
            out["entry"] = self.ex.create_order(
                pos.symbol, "limit", side, pos.units, pos.entry)
        except Exception as e:
            self.logger.error(f"[ccxt] entry order failed: {e}")
            return {"status": "failed", "error": str(e)}
        try:
            out["sl"] = self.ex.create_order(
                pos.symbol, "stop", opp, pos.units, None,
                params={"reduceOnly": True, "stopPrice": pos.stop_loss})
        except Exception as e:
            self.logger.warning(f"[ccxt] stop order not placed: {e}")
        try:
            out["tp"] = self.ex.create_order(
                pos.symbol, "take_profit", opp, pos.units, None,
                params={"reduceOnly": True, "stopPrice": pos.tp2})
        except Exception as e:
            self.logger.warning(f"[ccxt] take-profit order not placed: {e}")
        return {"status": "placed", "orders": out}


class OandaBroker(BrokerBase):
    def __init__(self, logger: logging.Logger, environment: str = "practice"):
        if _OandaAPI is None:
            raise ImportError("oandapyV20 is required for OandaBroker.")
        self.logger = logger
        token = os.getenv("OANDA_ACCESS_TOKEN")
        acct  = os.getenv("OANDA_ACCOUNT_ID")
        if not token or not acct:
            raise EnvironmentError("Set OANDA_ACCESS_TOKEN and OANDA_ACCOUNT_ID.")
        self.client = _OandaAPI(access_token=token, environment=environment)
        self.account_id = acct

    def get_equity(self) -> float:
        try:
            import oandapyV20.endpoints.accounts as accounts  # type: ignore
            req = accounts.AccountDetails(self.account_id)
            self.client.request(req)
            nav = req.response.get("account", {}).get("NAV")
            return float(nav) if nav is not None else 0.0
        except Exception as e:
            self.logger.error(f"[oanda] equity fetch failed: {e}")
            return 0.0

    def get_data_client(self):
        return self.client

    def place_bracket_order(self, pos: Position) -> dict[str, Any]:
        try:
            import oandapyV20.endpoints.orders as orders  # type: ignore
            units = pos.units if pos.side == "long" else -pos.units
            data = {
                "order": {
                    "type": "LIMIT",
                    "instrument": pos.symbol,
                    "units": str(units),
                    "price": str(pos.entry),
                    "timeInForce": "GTC",
                    "positionFill": "DEFAULT",
                    "stopLossOnFill": {"price": str(pos.stop_loss)},
                    "takeProfitOnFill": {"price": str(pos.tp2)},
                }
            }
            req = orders.OrderCreate(self.account_id, data=data)
            self.client.request(req)
            return {"status": "placed", "response": req.response}
        except Exception as e:
            self.logger.error(f"[oanda] order failed: {e}")
            return {"status": "failed", "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
#  Live bot
# ══════════════════════════════════════════════════════════════════════════════

class ICTLiveBot:
    def __init__(
        self,
        symbol: str,
        source: str,          # "ccxt" | "oanda" | "yfinance" | "local"
        broker: BrokerBase,
        risk: ICTRiskManager,
        logger: logging.Logger,
        exec_tf: str = "auto",  # "auto" | "5m" | "1h"
        local_dir: str = "gold_clean_data",
        local_prefix: str = "XAU",
    ):
        self.symbol       = symbol
        self.source       = source
        self.broker       = broker
        self.risk         = risk
        self.logger       = logger
        self.exec_tf      = exec_tf
        self.local_dir    = local_dir
        self.local_prefix = local_prefix
        self.notifier: TelegramNotifier | None = None

        self.positions:      list[Position]     = []
        self.pending_orders: list[PendingOrder] = []
        self.peak_equity: float = broker.get_equity() or 0.0
        self.last_cycle_bar: pd.Timestamp | None = None

        # Resolved execution TF (set on first cycle)
        self._effective_exec_tf: str = "5m"

    # ── Data / enrichment ─────────────────────────────────────────────────────

    def _resolve_exec_tf(self) -> str:
        if self.exec_tf != "auto":
            return self.exec_tf
        return "5m" if self.source == "ccxt" else "1h"

    def _fetch_raw_mtf(self) -> dict[str, pd.DataFrame]:
        """Fetch raw multi-timeframe OHLCV (unenriched)."""
        if self.source == "yfinance":
            return _fetch_yfinance_mtf(self.symbol)
        exchange = None if self.source == "local" else self.broker.get_data_client()
        return fetch_mtf(
            self.symbol, exchange,
            source=self.source,
            local_dir=self.local_dir,
            local_prefix=self.local_prefix,
        )

    def _enrich_cycle(self) -> dict[str, pd.DataFrame]:
        """
        Full enrichment pipeline — mirrors run_backtest_multi.run_asset() exactly:
          1. Fetch raw MTF data
          2. Resolve execution timeframe (5m for ccxt, 1h for yfinance)
          3. Enrich each needed TF with market_structure + dealing_range +
             liquidity + pd_arrays
          4. Forward-fill HTF levels onto exec frame + re-run sweeps + PD arrays
          5. Alias enriched["5m"] = enriched["1h"] when using 1h exec
        """
        mtf = self._fetch_raw_mtf()

        effective_exec_tf = self._resolve_exec_tf()
        self._effective_exec_tf = effective_exec_tf

        # pivot_n: smaller on exec frame → more CHoCH events
        pivot_n_exec = 3 if effective_exec_tf == "1h" else 5

        enriched: dict[str, pd.DataFrame] = {}
        for tf, df in mtf.items():
            if effective_exec_tf == "1h" and tf == "5m":
                continue  # skip 5m when 1h is the exec frame
            pn = pivot_n_exec if tf in ("5m", "1h") else 5
            df = add_market_structure(df, pivot_n=pn)
            df = compute_dealing_range(df, window="session")
            df = add_liquidity(df, lookback=20)
            df, _, _ = add_pd_arrays(df, timeframe=tf)
            enriched[tf] = df

        # Forward-fill HTF levels onto exec frame, re-run sweeps + PD arrays
        if effective_exec_tf == "1h":
            enriched["1h"] = _forward_fill_htf_onto_exec(enriched["4h"], enriched["1h"])
            enriched["5m"] = enriched["1h"]   # signal_engine always reads "5m" key
        else:
            enriched["5m"] = _forward_fill_htf_onto_exec(enriched["1h"], enriched["5m"])

        return enriched

    # ── Position management ───────────────────────────────────────────────────

    def _mark_to_market_and_manage(self, last_price: float) -> None:
        """
        Paper-mode position management:
          - TP1 hit → trail SL to breakeven
          - SL or TP2 hit → close position, update pnl + drawdown
        Live-mode: just logs (broker-side management is exchange-specific).
        """
        if not self.positions:
            return
        if isinstance(self.broker, PaperBroker):
            self.broker.set_last_price(self.symbol, last_price)

        still_open: list[Position] = []
        for p in self.positions:
            if p.status != "open":
                continue

            if not p.tp1_hit:
                if (p.side == "long"  and last_price >= p.tp1) or \
                   (p.side == "short" and last_price <= p.tp1):
                    p.tp1_hit    = True
                    p.stop_at_be = True
                    self.logger.info(
                        f"[pos] TP1 hit → trail SL to BE for {p.symbol} ({p.side})")
                    if self.notifier:
                        self.notifier.send(
                            f"TP1 hit: {p.symbol} {p.side} → trail SL to BE")

            effective_sl = p.entry if p.stop_at_be else p.stop_loss
            sl_hit  = (p.side == "long"  and last_price <= effective_sl) or \
                      (p.side == "short" and last_price >= effective_sl)
            tp2_hit = (p.side == "long"  and last_price >= p.tp2) or \
                      (p.side == "short" and last_price <= p.tp2)

            if sl_hit or tp2_hit:
                exit_price = effective_sl if sl_hit else p.tp2
                pnl = ((exit_price - p.entry) * p.units
                       if p.side == "long"
                       else (p.entry - exit_price) * p.units)
                p.status = "closed"
                self.logger.info(
                    f"[pos] CLOSED {p.symbol} {p.side} "
                    f"exit={exit_price:.5f} pnl={pnl:+.2f}")
                if self.notifier:
                    self.notifier.send(
                        f"Trade closed: {p.symbol} {p.side} "
                        f"exit={exit_price:.5f} pnl={pnl:+.2f}")

                self.risk.update_daily_pnl(pnl)
                eq = self.broker.get_equity()
                if isinstance(self.broker, PaperBroker):
                    self.broker.equity += pnl
                    eq = self.broker.get_equity()
                self.peak_equity = max(self.peak_equity, eq)
                if self.risk.check_drawdown(eq, self.peak_equity):
                    self.logger.warning(
                        "[risk] max drawdown reached → bot halted until manual reset")
                    if self.notifier:
                        self.notifier.send(
                            "RISK HALT: max drawdown reached (manual reset required)")

                self.risk.log_trade({
                    "symbol":    p.symbol,
                    "side":      p.side,
                    "units":     p.units,
                    "entry":     p.entry,
                    "exit":      exit_price,
                    "pnl":       pnl,
                    "tp1_hit":   p.tp1_hit,
                    "tp2":       p.tp2,
                    "stop_loss": p.stop_loss,
                    "opened_at": p.opened_at.isoformat(),
                    "closed_at": _utc_now().isoformat(),
                    "mode": "paper" if isinstance(self.broker, PaperBroker) else "live",
                })
            else:
                still_open.append(p)

        self.positions = still_open

    def _paper_fill_pending(self, last_bar: pd.Series) -> None:
        """
        Paper fill: a limit entry fills if the bar trades through the entry.
          long  fills if low  <= entry
          short fills if high >= entry
        """
        if not isinstance(self.broker, PaperBroker) or not self.pending_orders:
            return
        hi = float(last_bar.get("high"))
        lo = float(last_bar.get("low"))

        filled: list[PendingOrder] = []
        still:  list[PendingOrder] = []
        for o in self.pending_orders:
            if o.status != "pending":
                continue
            hit = (lo <= o.entry) if o.side == "long" else (hi >= o.entry)
            if hit:
                o.status = "filled"
                filled.append(o)
            else:
                still.append(o)
        self.pending_orders = still

        for o in filled:
            pos = Position(
                symbol=o.symbol, side=o.side, units=o.units,
                entry=o.entry, stop_loss=o.stop_loss,
                tp1=o.tp1, tp2=o.tp2, opened_at=_utc_now(),
            )
            self.positions.append(pos)
            self.logger.info(
                f"[PAPER] FILLED {o.symbol} {o.side} "
                f"units={o.units:.6f} entry={o.entry:.5f}")
            if self.notifier:
                self.notifier.send(
                    f"Order filled: {o.symbol} {o.side} entry={o.entry:.5f}")

    # ── Main cycle ────────────────────────────────────────────────────────────

    def run_cycle(self) -> None:
        """One trading cycle (5 min for ccxt, 1 h for yfinance)."""
        try:
            equity = self.broker.get_equity()
            if equity > 0:
                self.peak_equity = max(self.peak_equity, equity)
                if self.risk.check_drawdown(equity, self.peak_equity):
                    self.logger.warning("[risk] drawdown halt active → skipping cycle")
                    if self.notifier:
                        self.notifier.send(
                            "RISK HALT active: drawdown (manual reset required)")
                    return
                self.risk._roll_day_if_needed(equity=equity)
                if self.risk.halted_today:
                    self.logger.warning("[risk] daily halt active → skipping cycle")
                    if self.notifier:
                        self.notifier.send(
                            "RISK HALT active: max daily loss reached (halted for today)")
                    return

            enriched = self._enrich_cycle()

            df_exec = enriched["5m"]   # always "5m" key (aliased from "1h" if needed)
            last_bar_time = df_exec.index[-1]
            if (self.last_cycle_bar is not None
                    and last_bar_time <= self.last_cycle_bar):
                self.logger.info("[cycle] no new bar yet — skipping")
                return
            self.last_cycle_bar = last_bar_time

            last_price = float(df_exec["close"].iloc[-1])
            self._paper_fill_pending(df_exec.iloc[-1])
            self._mark_to_market_and_manage(last_price)

            # min_risk_atr matches backtest: tighter for 1h (ATR already large)
            min_risk_atr = 0.5 if self._effective_exec_tf == "1h" else 1.5

            sig = get_signal(
                enriched,
                equity=equity if equity > 0 else None,
                open_positions=len(self.positions),
                risk_manager=self.risk,
                idx=-1,
                min_conditions=MIN_CONDITIONS,   # 6-of-7 grace, matching backtest
            )

            self.logger.info(
                f"[cycle] {self.symbol} bar={last_bar_time} "
                f"close={last_price:.5f} exec={self._effective_exec_tf} "
                f"signal={sig.get('signal')} conds={sig.get('conditions_met')}/7 "
                f"rr={sig.get('rr_ratio')} kz={sig.get('kill_zone')} "
                f"approved={sig.get('approved')} reason={sig.get('approve_reason','')}"
            )

            if int(sig.get("signal", 0) or 0) == 0:
                return
            if sig.get("approved") is not True:
                return

            if self.notifier:
                self.notifier.send(
                    f"Signal: {self.symbol} "
                    f"{'LONG' if int(sig['signal'])==1 else 'SHORT'} "
                    f"entry={float(sig['entry_price']):.5f} "
                    f"sl={float(sig['stop_loss']):.5f} "
                    f"tp2={float(sig['tp2']):.5f} "
                    f"rr={float(sig['rr_ratio']):.2f} "
                    f"kz={sig.get('kill_zone')}"
                )

            side  = "long" if int(sig["signal"]) == 1 else "short"
            units = float(sig.get("position_units") or 0.0)
            if units <= 0:
                self.logger.info("[trade] blocked: position_units <= 0")
                return

            pos = Position(
                symbol=self.symbol, side=side, units=units,
                entry=float(sig["entry_price"]),
                stop_loss=float(sig["stop_loss"]),
                tp1=float(sig["tp1"]),
                tp2=float(sig["tp2"]),
                opened_at=_utc_now(),
            )

            if isinstance(self.broker, PaperBroker):
                order = PendingOrder(
                    symbol=self.symbol, side=side, units=units,
                    entry=pos.entry, stop_loss=pos.stop_loss,
                    tp1=pos.tp1, tp2=pos.tp2, created_at=_utc_now(),
                )
                self.pending_orders.append(order)
                self.logger.info(
                    f"[PAPER] Placed LIMIT {self.symbol} {side} "
                    f"units={units:.6f} entry={order.entry:.5f}")
                resp = {"mode": "paper", "status": "pending"}
            else:
                resp = self.broker.place_bracket_order(pos)
                self.logger.info(f"[trade] placed: {resp}")
                if resp.get("status") == "placed":
                    self.positions.append(pos)

            if self.notifier:
                self.notifier.send(
                    f"Trade opened: {self.symbol} {side} units={units:.6f} "
                    f"(paper={isinstance(self.broker, PaperBroker)})")

            self.risk.log_trade({
                "symbol":    self.symbol,
                "side":      side,
                "units":     units,
                "entry":     float(sig["entry_price"]),
                "stop_loss": float(sig["stop_loss"]),
                "tp1":       float(sig["tp1"]),
                "tp2":       float(sig["tp2"]),
                "rr_ratio":  float(sig["rr_ratio"]),
                "kill_zone": str(sig.get("kill_zone")),
                "confluence": ",".join(sig.get("confluence") or []),
                "exec_tf":   self._effective_exec_tf,
                "mode": "paper" if isinstance(self.broker, PaperBroker) else "live",
                "event": "opened",
            })

        except Exception as e:
            self.logger.exception(f"[cycle] error: {e}")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI helpers
# ══════════════════════════════════════════════════════════════════════════════

def _build_broker(args: argparse.Namespace, logger: logging.Logger) -> BrokerBase:
    if args.paper or args.source == "yfinance":
        if args.source == "yfinance" and not args.paper:
            logger.info("[broker] yfinance source → paper-only mode (no live broker)")
        return PaperBroker(initial_equity=float(args.paper_equity), logger=logger)
    if args.source == "ccxt":
        return CCXTBroker(
            exchange_id=args.exchange_id, logger=logger,
            allow_unsafe_brackets=bool(args.allow_unsafe_live))
    if args.source == "oanda":
        return OandaBroker(logger=logger, environment=args.oanda_env)
    raise ValueError(
        "[bot] live mode requires --source ccxt or --source oanda "
        "(or use --paper / --source yfinance).")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="ICT bot live trading loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # Paper trade crypto (Binance, 5m exec)
  python bot.py --symbol BTC/USDT --paper

  # Paper trade gold futures (yfinance, 1h exec, auto-paper)
  python bot.py --symbol GC=F --source yfinance --paper

  # Paper trade EUR/USD (yfinance, 1h exec)
  python bot.py --symbol EURUSD=X --source yfinance --paper

  # Live ccxt + Telegram alerts
  python bot.py --symbol BTC/USDT --source ccxt --telegram

  # Single cycle (no scheduler)
  python bot.py --symbol BTC/USDT --paper --run-once

Sources
-------
  ccxt      — Binance (or other ccxt exchange), 5m exec, supports live orders
  yfinance  — Free delayed data, 1h exec, paper-only
  oanda     — OANDA forex, 5m exec, requires OANDA_ACCESS_TOKEN / OANDA_ACCOUNT_ID
  local     — Offline CSV (backtest-only, paper mode)
""",
    )
    ap.add_argument("--symbol",         required=True,
                    help="e.g. BTC/USDT (ccxt) | EURUSD=X (yfinance) | EUR_USD (oanda)")
    ap.add_argument("--source",         default="ccxt",
                    choices=["ccxt", "oanda", "yfinance", "local"],
                    help="data source (default: ccxt)")
    ap.add_argument("--exec-tf",        default="auto", choices=["auto", "5m", "1h"],
                    help="execution timeframe (default: auto → 5m for ccxt, 1h for yfinance)")
    ap.add_argument("--exchange-id",    default="binance",
                    help="ccxt exchange id (default: binance)")
    ap.add_argument("--oanda-env",      default="practice",
                    choices=["practice", "live"], help="oanda environment")
    ap.add_argument("--paper",          action="store_true",
                    help="paper trading (no real orders)")
    ap.add_argument("--paper-equity",   type=float, default=10_000.0,
                    help="starting equity for paper mode (default: 10000)")
    ap.add_argument("--local-dir",      default="gold_clean_data",
                    help="CSV directory for --source local")
    ap.add_argument("--local-prefix",   default="XAU",
                    help="CSV filename prefix for --source local")
    ap.add_argument("--log",            default="bot.log",
                    help="log file path (default: bot.log)")
    ap.add_argument("--run-once",       action="store_true",
                    help="run one cycle and exit (no scheduler)")
    ap.add_argument("--telegram",       action="store_true",
                    help="enable Telegram alerts (requires TELEGRAM_TOKEN + TELEGRAM_CHAT_ID)")
    ap.add_argument("--allow-unsafe-live", action="store_true",
                    help="attempt generic ccxt bracket orders (exchange-specific, use with caution)")
    args = ap.parse_args()

    logger = _setup_logger(args.log)
    pd.set_option("future.no_silent_downcasting", True)
    logger.info(
        f"[boot] symbol={args.symbol} source={args.source} "
        f"exec_tf={args.exec_tf} paper={args.paper}")

    broker = _build_broker(args, logger)
    risk   = ICTRiskManager()

    bot = ICTLiveBot(
        symbol=args.symbol,
        source=args.source,
        broker=broker,
        risk=risk,
        logger=logger,
        exec_tf=args.exec_tf,
        local_dir=args.local_dir,
        local_prefix=args.local_prefix,
    )

    if args.telegram:
        token   = os.getenv("TELEGRAM_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            raise EnvironmentError(
                "Set TELEGRAM_TOKEN and TELEGRAM_CHAT_ID to use --telegram.")
        bot.notifier = TelegramNotifier(token=token, chat_id=chat_id, logger=logger)
        logger.info("[telegram] enabled")

    if args.run_once:
        bot.run_cycle()
        return

    if not _HAS_APSCHEDULER:
        raise ImportError(
            "APScheduler is required for scheduled mode. "
            "Install it with: pip install APScheduler  (or run with --run-once).")

    # Schedule: every 5 min for ccxt/oanda/5m, every hour for yfinance/1h
    scheduler = BackgroundScheduler(timezone="UTC")
    exec_tf_resolved = bot._resolve_exec_tf()
    if exec_tf_resolved == "1h":
        trigger = CronTrigger(minute=5, second=0)   # 5 seconds past the hour
        logger.info("[sched] 1h exec → running at :05 of every hour")
    else:
        trigger = CronTrigger(minute="*/5", second=5)
        logger.info("[sched] 5m exec → running every 5 minutes at +5s")

    scheduler.add_job(
        bot.run_cycle, trigger=trigger,
        id="ict_cycle", max_instances=1, coalesce=True)
    scheduler.start()

    next_run = _next_5m_close(_utc_now()).isoformat()
    logger.info(f"[sched] started. first run ~{next_run}Z")

    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        logger.info("[sched] stopping …")
        scheduler.shutdown(wait=False)


if __name__ == "__main__":
    main()
