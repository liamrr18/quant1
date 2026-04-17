"""Futures backtesting engine with realistic execution assumptions.

KEY DIFFERENCES FROM EQUITY ENGINE:
- Position sizing: contracts instead of shares, based on risk-per-trade
  and stop distance, not capital percentage
- P&L calculation: uses futures multiplier (e.g., $50/SPY point for MES)
- Cost model: fixed per-contract costs (tick slippage + commission)
  instead of per-share slippage/commission
- Margin tracking: ensures sufficient margin before entry
- No fractional contracts

Execution model (same as equity):
- Signals evaluated on bar N, execution on bar N+1 open
- No lookahead bias
- Conservative slippage (1 tick per side)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from trading.strategies.base import Strategy, Trade
from trading.data.contracts import (
    FuturesContract, CONTRACTS,
    total_cost_per_contract, slippage_cost_per_contract,
    commission_cost_per_contract,
)
from trading.config import (
    INITIAL_CAPITAL, MAX_RISK_PER_TRADE_PCT,
    MAX_CONTRACTS_MES, MAX_CONTRACTS_MNQ,
    MARGIN_SAFETY_MULTIPLE, STOP_LOSS_PCT, TAKE_PROFIT_PCT,
)

log = logging.getLogger(__name__)

MAX_CONTRACTS = {"MES": MAX_CONTRACTS_MES, "MNQ": MAX_CONTRACTS_MNQ}


@dataclass
class BacktestResult:
    """Complete backtest output for futures."""
    strategy_name: str
    symbol: str
    params: dict
    contract: FuturesContract
    trades: list[Trade]
    equity_curve: pd.Series
    daily_returns: pd.Series
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_trade_dollars: float
    avg_trade_pct: float
    num_trades: int
    avg_bars_held: float
    exposure_pct: float
    total_pnl: float
    avg_contracts: float
    total_costs: float
    start_date: str
    end_date: str


def calculate_contracts(capital: float, entry_price: float, stop_price: float,
                        contract: FuturesContract, futures_symbol: str,
                        risk_pct: float = MAX_RISK_PER_TRADE_PCT) -> int:
    """Calculate number of contracts based on risk-per-trade.

    contracts = min(
        max_contracts_cap,
        floor(risk_dollars / risk_per_contract),
        floor(available_margin / margin_per_contract)
    )

    where risk_per_contract = |entry - stop| * multiplier
    """
    if entry_price <= 0 or stop_price <= 0 or capital <= 0:
        return 0

    risk_dollars = capital * risk_pct
    stop_distance = abs(entry_price - stop_price)

    if stop_distance <= 0:
        return 0

    # Dollar risk per contract = stop distance in ETF price * multiplier
    risk_per_contract = stop_distance * contract.multiplier
    if risk_per_contract <= 0:
        return 0

    # Add round-trip costs to risk calculation
    cost_per_contract = total_cost_per_contract(contract)
    effective_risk = risk_per_contract + cost_per_contract

    contracts_by_risk = int(risk_dollars / effective_risk)

    # Margin constraint
    available_margin = capital / MARGIN_SAFETY_MULTIPLE
    contracts_by_margin = int(available_margin / contract.margin_intraday)

    # Hard cap
    max_cap = MAX_CONTRACTS.get(futures_symbol, 30)

    contracts = min(contracts_by_risk, contracts_by_margin, max_cap)
    return max(contracts, 0)


def run_backtest(df: pd.DataFrame, strategy: Strategy, futures_symbol: str,
                 initial_capital: float = INITIAL_CAPITAL,
                 contract: FuturesContract = None,
                 risk_pct: float = MAX_RISK_PER_TRADE_PCT) -> BacktestResult:
    """Run a futures backtest.

    The signal column should already be in df (from strategy.generate_signals).
    Execution happens on the NEXT bar's open after a signal change.

    P&L is calculated using the futures multiplier:
        pnl = (exit_price - entry_price) * multiplier * contracts - costs

    where prices are in ETF space (SPY/QQQ) and multiplier converts to
    futures dollar P&L (MES: $50/SPY point, MNQ: $80/QQQ point).
    """
    if contract is None:
        contract = CONTRACTS[futures_symbol]

    if "signal" not in df.columns:
        raise ValueError("DataFrame must have 'signal' column")

    df = df.copy().reset_index(drop=True)
    n = len(df)

    # State tracking
    capital = initial_capital
    position = 0        # contracts held (positive=long, negative=short)
    entry_price = 0.0
    stop_price = 0.0
    entry_time = None
    trades: list[Trade] = []
    equity = np.full(n, initial_capital)
    bars_in_position = 0
    total_contracts_traded = 0
    total_costs = 0.0

    # Per-contract costs
    cost_per_contract_rt = total_cost_per_contract(contract)
    slippage_in_etf = contract.tick_value * contract.slippage_ticks / contract.multiplier

    prev_signal = 0
    current_date = None

    for i in range(1, n):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        prev_signal = int(prev_row["signal"])
        exec_price = float(row["open"])
        date = row["date"]

        eod_close = row["minute_of_day"] >= 15 * 60 + 50
        new_day = date != current_date
        if new_day:
            current_date = date

        # ---- Check exits first ----
        if position != 0:
            exit_reason = None

            if eod_close or new_day:
                exit_reason = "eod"
            elif position > 0 and exec_price <= entry_price * (1 - STOP_LOSS_PCT):
                exit_reason = "stop"
            elif position < 0 and exec_price >= entry_price * (1 + STOP_LOSS_PCT):
                exit_reason = "stop"
            elif position > 0 and exec_price >= entry_price * (1 + TAKE_PROFIT_PCT):
                exit_reason = "target"
            elif position < 0 and exec_price <= entry_price * (1 - TAKE_PROFIT_PCT):
                exit_reason = "target"
            elif prev_signal == 0 and position != 0:
                exit_reason = "signal"
            elif prev_signal != 0 and np.sign(prev_signal) != np.sign(position):
                exit_reason = "signal"

            if exit_reason:
                contracts = abs(position)

                # Apply tick slippage to exit
                if position > 0:
                    fill_price = exec_price - slippage_in_etf
                else:
                    fill_price = exec_price + slippage_in_etf

                # P&L using futures multiplier
                if position > 0:
                    raw_pnl = (fill_price - entry_price) * contract.multiplier * contracts
                else:
                    raw_pnl = (entry_price - fill_price) * contract.multiplier * contracts

                # Costs: commission only (slippage already in fill_price)
                exit_commission = contracts * contract.commission_per_side
                trade_cost = exit_commission  # Entry commission already deducted
                raw_pnl -= exit_commission
                total_costs += exit_commission

                pnl = raw_pnl
                pnl_pct = pnl / (capital) if capital > 0 else 0
                capital += pnl

                trades.append(Trade(
                    symbol=futures_symbol,
                    entry_time=entry_time,
                    exit_time=row["dt"],
                    direction="long" if position > 0 else "short",
                    entry_price=entry_price,
                    exit_price=fill_price,
                    contracts=contracts,
                    pnl=pnl,
                    pnl_pct=pnl_pct,
                    exit_reason=exit_reason,
                ))

                position = 0
                entry_price = 0.0

        # ---- Check entries ----
        if position == 0 and not eod_close and prev_signal != 0:
            if not new_day:
                # Get stop level from the previous bar's strategy state
                or_high = float(prev_row.get("or_high", 0)) if pd.notna(prev_row.get("or_high", 0)) else 0
                or_low = float(prev_row.get("or_low", 0)) if pd.notna(prev_row.get("or_low", 0)) else 0

                if or_high > 0 and or_low > 0:
                    if prev_signal > 0:
                        stop_price = or_low
                        entry_with_slip = exec_price + slippage_in_etf
                    else:
                        stop_price = or_high
                        entry_with_slip = exec_price - slippage_in_etf

                    contracts = calculate_contracts(
                        capital, entry_with_slip, stop_price,
                        contract, futures_symbol, risk_pct
                    )

                    if contracts > 0:
                        entry_price = entry_with_slip

                        # Entry commission
                        entry_commission = contracts * contract.commission_per_side
                        capital -= entry_commission
                        total_costs += entry_commission

                        position = contracts if prev_signal > 0 else -contracts
                        entry_time = row["dt"]
                        total_contracts_traded += contracts

        # Track equity
        if position != 0:
            mark = float(row["close"])
            if position > 0:
                unrealized = (mark - entry_price) * contract.multiplier * abs(position)
            else:
                unrealized = (entry_price - mark) * contract.multiplier * abs(position)
            equity[i] = capital + unrealized
            bars_in_position += 1
        else:
            equity[i] = capital

    # Close any remaining position at last bar
    if position != 0:
        last = df.iloc[-1]
        contracts = abs(position)
        fill = float(last["close"]) - slippage_in_etf * np.sign(position)

        if position > 0:
            pnl = (fill - entry_price) * contract.multiplier * contracts
        else:
            pnl = (entry_price - fill) * contract.multiplier * contracts

        exit_comm = contracts * contract.commission_per_side
        pnl -= exit_comm
        total_costs += exit_comm
        capital += pnl

        trades.append(Trade(
            symbol=futures_symbol, entry_time=entry_time, exit_time=last["dt"],
            direction="long" if position > 0 else "short",
            entry_price=entry_price, exit_price=fill, contracts=contracts,
            pnl=pnl, pnl_pct=pnl / capital if capital > 0 else 0,
            exit_reason="eod",
        ))
        equity[-1] = capital

    equity_series = pd.Series(equity, index=df["dt"])
    metrics = compute_metrics(trades, equity_series, initial_capital, n,
                              bars_in_position, total_contracts_traded)

    return BacktestResult(
        strategy_name=strategy.name,
        symbol=futures_symbol,
        params=strategy.get_params(),
        contract=contract,
        trades=trades,
        equity_curve=equity_series,
        daily_returns=metrics["daily_returns"],
        total_return=metrics["total_return"],
        annual_return=metrics["annual_return"],
        sharpe_ratio=metrics["sharpe_ratio"],
        max_drawdown=metrics["max_drawdown"],
        win_rate=metrics["win_rate"],
        profit_factor=metrics["profit_factor"],
        avg_trade_dollars=metrics["avg_trade_dollars"],
        avg_trade_pct=metrics["avg_trade_pct"],
        num_trades=metrics["num_trades"],
        avg_bars_held=metrics["avg_bars_held"],
        exposure_pct=metrics["exposure_pct"],
        total_pnl=metrics["total_pnl"],
        avg_contracts=metrics["avg_contracts"],
        total_costs=total_costs,
        start_date=str(df["date"].iloc[0]),
        end_date=str(df["date"].iloc[-1]),
    )


def compute_metrics(trades: list[Trade], equity: pd.Series,
                    initial_capital: float, total_bars: int,
                    bars_in_position: int, total_contracts: int) -> dict:
    """Compute performance metrics from trade list and equity curve."""
    num_trades = len(trades)
    final_equity = float(equity.iloc[-1])
    total_return = (final_equity / initial_capital) - 1
    total_pnl = final_equity - initial_capital

    # Daily returns for Sharpe
    daily_equity = equity.resample("D").last().dropna()
    daily_returns = daily_equity.pct_change().dropna()

    n_days = max(len(daily_equity), 1)
    annual_return = (1 + total_return) ** (252 / n_days) - 1 if n_days > 1 else 0

    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    else:
        sharpe_ratio = 0.0

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    if num_trades > 0:
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / num_trades

        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 0.001
        profit_factor = gross_profit / gross_loss

        avg_trade_dollars = np.mean([t.pnl for t in trades])
        avg_trade_pct = np.mean([t.pnl_pct for t in trades]) * 100

        avg_bars_held = np.mean([
            max(1, (t.exit_time - t.entry_time).total_seconds() / 60)
            for t in trades
        ])
        avg_contracts = total_contracts / num_trades if num_trades > 0 else 0
    else:
        win_rate = 0
        profit_factor = 0
        avg_trade_dollars = 0
        avg_trade_pct = 0
        avg_bars_held = 0
        avg_contracts = 0

    exposure_pct = (bars_in_position / total_bars * 100) if total_bars > 0 else 0

    return {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_trade_dollars": avg_trade_dollars,
        "avg_trade_pct": avg_trade_pct,
        "num_trades": num_trades,
        "avg_bars_held": avg_bars_held,
        "exposure_pct": exposure_pct,
        "daily_returns": daily_returns,
        "total_pnl": total_pnl,
        "avg_contracts": avg_contracts,
    }
