"""Core trading types shared by strategies, decision engine, sizing, risk, broker and portfolio.

Slotted dataclasses rather than pydantic models: they are created in the backtest loop, so
construction cost matters. Quantities are floats so markets with fractional shares fit the same
types; whole-share markets simply hold whole numbers. Money is in the market's currency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import StrEnum

from trader.market.base import CostBreakdown


class Side(StrEnum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(StrEnum):
    MARKET_ON_OPEN = "MOO"
    STOP = "STOP"
    TARGET = "TARGET"


class ExitReason(StrEnum):
    STOP = "STOP"
    TRAIL_STOP = "TRAIL_STOP"
    TARGET = "TARGET"
    RULE = "RULE"
    TIME = "TIME"
    STAGNATION = "STAGNATION"
    REGIME = "REGIME"
    DELIST = "DELIST"
    MANUAL = "MANUAL"


class Regime(StrEnum):
    RISK_ON = "RISK_ON"
    NEUTRAL = "NEUTRAL"
    RISK_OFF = "RISK_OFF"


class DecisionKind(StrEnum):
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    HOLD = "HOLD"
    SKIP = "SKIP"
    REJECT = "REJECT"
    PAUSE = "PAUSE"
    NOOP = "NOOP"


class Stage(StrEnum):
    SIGNAL = "signal"
    DECISION = "decision"
    SIZING = "sizing"
    RISK = "risk"
    ORDER = "order"
    FILLED = "filled"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class Bar:
    date: date
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    prev_close: float | None = None
    unadj_close: float | None = None  # price as actually traded that day

    @property
    def traded_scale(self) -> float:
        """Multiply this bar's adjusted prices by this to get the prices actually traded that day
        (e.g. 100 for a stock whose later 1:100 split makes its 2015 price look 100 times smaller)."""
        if self.unadj_close and self.close and math.isfinite(self.unadj_close):
            return self.unadj_close / self.close
        return 1.0


@dataclass(frozen=True, slots=True)
class ExitSpec:
    """Exit policy a strategy attaches to the positions it opens."""

    atr_mult_init: float
    trail_atr_mult: float | None = None
    target_r: float | None = None
    max_bars: int | None = None
    stagnation_bars: int | None = None
    stagnation_min_r: float | None = None
    rule_exit: bool = False


@dataclass(frozen=True, slots=True)
class Signal:
    """A strategy's output at the close of `date`; long-only."""

    strategy: str
    symbol: str
    date: date
    kind: str  # "ENTRY" or "EXIT"
    ref_price: float  # close on `date`
    atr: float
    strength: float  # 0..1, a deterministic function of features
    stop_price: float | None = None
    target_price: float | None = None
    exit_spec: ExitSpec | None = None
    exit_reason: ExitReason | None = None
    rationale: tuple[str, ...] = ()
    features: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True, slots=True)
class EventAssessment:
    """LLM news assessment for one symbol and day; usable only once `available_at` has passed."""

    symbol: str
    date: date
    available_at: datetime
    direction: int  # -1, 0, +1
    severity: int  # 0..3
    is_material: bool
    confidence: float  # 0..1
    event_type: str | None = None
    summary: str = ""


@dataclass(frozen=True, slots=True)
class GateResult:
    would_veto: bool
    penalty: float
    reason: str


@dataclass(frozen=True, slots=True)
class Opportunity:
    symbol: str
    date: date
    primary_strategy: str
    strategies: tuple[str, ...]
    strength: float
    score: float
    components: tuple[tuple[str, float], ...]
    signal: Signal
    rank: int | None = None
    status: str = "SELECTED"  # or "SKIPPED"
    skip_reason: str | None = None
    llm: GateResult | None = None
    sector: str | None = None


@dataclass(frozen=True, slots=True)
class SizingResult:
    quantity: float
    est_entry: float
    position_value: float
    target_risk: float
    actual_risk: float
    est_round_trip_cost: float
    est_cost_pct: float
    binding: str  # risk | position_cap | cash | none
    ok: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class TradeProposal:
    """What the decision layer wants to do. The risk engine decides whether it may."""

    proposal_id: str
    date: date
    effective_date: date
    symbol: str
    side: Side
    strategy: str
    quantity: float
    ref_price: float
    est_entry: float
    stop_price: float | None = None
    target_price: float | None = None
    exit_spec: ExitSpec | None = None
    sizing: SizingResult | None = None
    score: float | None = None
    rationale: tuple[str, ...] = ()
    exit_reason: ExitReason | None = None


@dataclass(frozen=True, slots=True)
class RuleResult:
    rule: str
    passed: bool
    detail: str
    adjusted_quantity: float | None = None


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    approved: bool
    quantity: float
    checks: tuple[RuleResult, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class Order:
    order_id: str
    created: date
    effective_date: date
    symbol: str
    side: Side
    quantity: float
    order_type: OrderType
    strategy: str
    proposal_id: str
    trigger_price: float | None = None
    max_gap_pct: float | None = None
    attempts: int = 0


@dataclass(frozen=True, slots=True)
class Fill:
    fill_id: str
    order_id: str
    date: date
    symbol: str
    side: Side
    quantity: float
    raw_price: float  # open or trigger price before slippage
    price: float  # after slippage and tick rounding
    slippage: float  # money lost to slippage on this fill
    costs: CostBreakdown
    gross_value: float
    net_cash_delta: float  # negative for buys, positive for sells, costs included
    fill_kind: str  # open | stop_gap | stop_intraday | target_gap | target_intraday | delist
    strategy: str
    exit_reason: ExitReason | None = None


@dataclass(slots=True)
class Position:
    """An open holding. Mutable: stops trail and counters advance each day."""

    symbol: str
    strategy: str
    quantity: float
    avg_price: float
    entry_date: date
    entry_fill_id: str
    entry_costs: float
    stop_price: float
    initial_stop: float
    exit_spec: ExitSpec
    stop_active_from: date
    target_price: float | None = None
    highest_close: float = 0.0
    bars_held: int = 0
    last_close: float = 0.0
    exit_pending: bool = False
    sector: str | None = None
    opportunity_score: float | None = None


@dataclass(frozen=True, slots=True)
class Trade:
    """A completed round trip."""

    trade_id: str
    symbol: str
    strategy: str
    entry_date: date
    exit_date: date
    quantity: float
    entry_price: float
    exit_price: float
    gross_pnl: float
    costs: float
    net_pnl: float
    r_multiple: float
    initial_risk: float
    holding_days: int
    exit_reason: ExitReason
    mae_pct: float = 0.0
    mfe_pct: float = 0.0
    sector: str | None = None
    strategies_confluent: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EquityPoint:
    date: date
    cash_settled: float
    cash_unsettled: float
    positions_value: float
    equity: float
    n_positions: int
    exposure_pct: float
    unrealized: float
    realized_cum: float
    costs_cum: float
    peak: float
    drawdown_pct: float
    regime: Regime
