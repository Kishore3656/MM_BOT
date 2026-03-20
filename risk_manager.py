"""
risk_manager.py
===============
Portfolio-level risk controls for the ICT bot.

This module is intentionally broker-agnostic: it approves/rejects trades and
computes a position size in "units" based on stop distance and equity risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import csv
from typing import Any


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class ICTRiskManager:
    # Hard risk caps
    max_risk_per_trade: float = 0.01          # 1% of equity
    max_daily_loss: float = 0.03              # 3% of day's starting equity
    max_drawdown: float = 0.10                # 10% from equity peak
    max_concurrent_positions: int = 2

    # Trade-quality filters
    min_rr_ratio: float = 2.0
    kill_zone_only: bool = True

    # State
    halted_today: bool = False
    drawdown_halt: bool = False
    daily_start_equity: float | None = None
    daily_pnl: float = 0.0
    current_day_utc: str | None = None  # YYYY-MM-DD

    trades_path: Path = field(default_factory=lambda: Path(__file__).with_name("trades.csv"))

    def _roll_day_if_needed(self, equity: float | None = None) -> None:
        day = _utc_now().date().isoformat()
        if self.current_day_utc != day:
            self.current_day_utc = day
            self.halted_today = False
            self.daily_pnl = 0.0
            if equity is not None:
                self.daily_start_equity = float(equity)

    def approve_trade(
        self,
        signal_dict: dict[str, Any],
        equity: float,
        open_positions: list[dict[str, Any]] | int,
    ) -> tuple[bool, str]:
        """
        Return (approved, reason). Reason is "approved" on success.

        Expects signal_dict similar to signal_engine.score_signal() output:
          - signal (1/-1/0)
          - rr_ratio (float)
          - kill_zone (str)
          - entry_price (float), stop_loss (float) if sizing is needed
        """
        equity = float(equity)
        self._roll_day_if_needed(equity=equity)

        if self.drawdown_halt:
            return False, "halted: max drawdown reached (manual reset required)"
        if self.halted_today:
            return False, "halted: max daily loss reached"

        sig = int(signal_dict.get("signal", 0) or 0)
        if sig == 0:
            return False, "no-trade signal"

        if self.kill_zone_only:
            kz = str(signal_dict.get("kill_zone", "") or "")
            if kz.lower() not in ("london", "newyork", "nypm"):
                return False, "blocked: outside kill zones"

        rr = signal_dict.get("rr_ratio", None)
        if rr is None:
            return False, "blocked: missing rr_ratio"
        if float(rr) < float(self.min_rr_ratio):
            return False, f"blocked: rr_ratio<{self.min_rr_ratio:g}"

        if isinstance(open_positions, int):
            n_open = int(open_positions)
        else:
            n_open = len(open_positions)
        if n_open >= int(self.max_concurrent_positions):
            return False, f"blocked: max_concurrent_positions={self.max_concurrent_positions}"

        # Ensure the stop distance is valid (avoids divide-by-zero sizing)
        entry = signal_dict.get("entry_price", None)
        sl = signal_dict.get("stop_loss", None)
        if entry is None or sl is None:
            return False, "blocked: missing entry/stop_loss"
        if abs(float(entry) - float(sl)) <= 0.0:
            return False, "blocked: zero stop distance"

        return True, "approved"

    def compute_position_size(self, equity: float, entry: float, stop_loss: float) -> float:
        """
        Units sized so that max loss at stop equals max_risk_per_trade * equity.
        """
        equity = float(equity)
        entry = float(entry)
        stop_loss = float(stop_loss)
        risk_per_unit = abs(entry - stop_loss)
        if risk_per_unit <= 0:
            return 0.0
        risk_budget = equity * float(self.max_risk_per_trade)
        return float(risk_budget / risk_per_unit)

    def update_daily_pnl(self, pnl: float) -> bool:
        """
        Add pnl (can be negative) and halt for the day if limit breached.
        Returns True if bot is halted_today after the update.
        """
        self._roll_day_if_needed()
        self.daily_pnl += float(pnl)
        if self.daily_start_equity is None:
            # If we don't know starting equity, we can only track raw pnl.
            return self.halted_today
        if self.daily_pnl <= -float(self.max_daily_loss) * float(self.daily_start_equity):
            self.halted_today = True
        return self.halted_today

    def check_drawdown(self, current_equity: float, peak_equity: float) -> bool:
        """
        Halt until manual reset if drawdown >= max_drawdown from peak.
        Returns True if drawdown_halt is active.
        """
        current_equity = float(current_equity)
        peak_equity = float(peak_equity)
        if peak_equity <= 0:
            return self.drawdown_halt
        dd = (peak_equity - current_equity) / peak_equity
        if dd >= float(self.max_drawdown):
            self.drawdown_halt = True
        return self.drawdown_halt

    def manual_reset(self) -> None:
        """
        Manual operator reset for drawdown_halt (and day halt if desired).
        """
        self.drawdown_halt = False
        self.halted_today = False

    def log_trade(self, trade_dict: dict[str, Any]) -> None:
        """
        Append a trade dict to trades.csv with an injected UTC timestamp.
        """
        row = dict(trade_dict)
        row.setdefault("timestamp_utc", _utc_now().isoformat())

        path = Path(self.trades_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Stable header: union of existing header and new keys
        if path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                existing_header = next(reader, [])
        else:
            existing_header = []

        keys = list(existing_header)
        for k in row.keys():
            if k not in keys:
                keys.append(k)

        write_header = (not path.exists()) or (keys != existing_header)

        # If header changed, rewrite file with new header (preserving old rows)
        if write_header and path.exists():
            with path.open("r", newline="", encoding="utf-8") as f:
                old_rows = list(csv.DictReader(f))
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for r in old_rows:
                    w.writerow({k: r.get(k, "") for k in keys})
                w.writerow({k: row.get(k, "") for k in keys})
            return

        with path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            if not path.exists() or (not existing_header):
                w.writeheader()
            w.writerow({k: row.get(k, "") for k in keys})

