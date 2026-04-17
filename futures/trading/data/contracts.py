"""Micro futures contract specifications.

MES (Micro E-mini S&P 500) and MNQ (Micro E-mini Nasdaq 100) are
quarter-sized versions of the E-mini contracts. They track the same
indices as SPY and QQQ respectively, allowing us to use ETF data as
a proxy for backtesting during regular trading hours (09:30-16:00 ET).

PROXY ASSUMPTION:
  MES tracks SPY * ~10 (S&P 500 index level) tick-for-tick during RTH.
  MNQ tracks QQQ * ~40 (Nasdaq 100 index level) tick-for-tick during RTH.
  Divergence risk exists for:
    - Overnight/extended hours (futures trade nearly 24h, ETFs don't)
    - Futures premium/discount (basis) during high-volatility periods
    - Roll dates around quarterly expiry
  Since our ORB strategy trades only 09:45-15:30 ET, the proxy is
  highly accurate. Basis risk is negligible for intraday holding periods.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FuturesContract:
    """Specification for a micro futures contract."""
    symbol: str              # Futures symbol (MES, MNQ)
    name: str                # Human-readable name
    proxy_symbol: str        # ETF used as data proxy (SPY, QQQ)
    point_value: float       # Dollar value per index point
    tick_size: float          # Minimum price increment (index points)
    tick_value: float         # Dollar value per tick
    etf_to_index: float      # Approximate ETF-to-index ratio
    multiplier: float         # P&L per $1 ETF move per contract
    margin_intraday: float   # Intraday margin per contract ($)
    commission_per_side: float  # Exchange+clearing+NFA per side per contract ($)
    slippage_ticks: int       # Assumed slippage in ticks per side


MES = FuturesContract(
    symbol="MES",
    name="Micro E-mini S&P 500",
    proxy_symbol="SPY",
    point_value=5.0,           # $5 per S&P 500 index point
    tick_size=0.25,            # 0.25 index points
    tick_value=1.25,           # $5 * 0.25 = $1.25 per tick
    etf_to_index=10.0,        # SPY * 10 ~ S&P 500
    multiplier=50.0,           # $5 * 10 = $50 per $1 SPY move per contract
    margin_intraday=1500.0,    # ~$1,500 intraday margin
    commission_per_side=0.31,  # $0.62 round trip ($0.31 each way)
    slippage_ticks=1,          # 1 tick per side (conservative)
)

MNQ = FuturesContract(
    symbol="MNQ",
    name="Micro E-mini Nasdaq 100",
    proxy_symbol="QQQ",
    point_value=2.0,           # $2 per Nasdaq 100 index point
    tick_size=0.25,            # 0.25 index points
    tick_value=0.50,           # $2 * 0.25 = $0.50 per tick
    etf_to_index=40.0,        # QQQ * 40 ~ Nasdaq 100
    multiplier=80.0,           # $2 * 40 = $80 per $1 QQQ move per contract
    margin_intraday=1800.0,    # ~$1,800 intraday margin
    commission_per_side=0.31,  # $0.62 round trip ($0.31 each way)
    slippage_ticks=1,          # 1 tick per side (conservative)
)

MGC = FuturesContract(
    symbol="MGC",
    name="Micro Gold Futures",
    proxy_symbol="GLD",
    point_value=10.0,          # $10 per gold point
    tick_size=0.10,            # 0.10 points
    tick_value=1.00,           # $10 * 0.10 = $1.00 per tick
    etf_to_index=1.0,         # GLD ≈ 1/10 of gold price (not used for MGC)
    multiplier=10.0,           # $10 per point
    margin_intraday=1000.0,    # ~$1,000 intraday margin
    commission_per_side=0.31,  # $0.62 round trip
    slippage_ticks=1,          # 1 tick per side
)

# Registry for lookup
CONTRACTS = {
    "MES": MES,
    "MNQ": MNQ,
    "MGC": MGC,
}

# Quarterly expiry months (Mar, Jun, Sep, Dec)
ROLL_MONTHS = [3, 6, 9, 12]


def slippage_cost_per_contract(contract: FuturesContract, sides: int = 2) -> float:
    """Total slippage cost per contract for entry+exit (both sides)."""
    return contract.tick_value * contract.slippage_ticks * sides


def commission_cost_per_contract(contract: FuturesContract, sides: int = 2) -> float:
    """Total commission cost per contract for entry+exit (both sides)."""
    return contract.commission_per_side * sides


def total_cost_per_contract(contract: FuturesContract) -> float:
    """Total round-trip cost per contract (slippage + commission)."""
    return slippage_cost_per_contract(contract) + commission_cost_per_contract(contract)


def etf_move_to_futures_pnl(etf_price_change: float, contract: FuturesContract,
                             num_contracts: int) -> float:
    """Convert ETF price change to futures P&L (before costs)."""
    return etf_price_change * contract.multiplier * num_contracts


def futures_stop_distance_dollars(stop_distance_etf: float,
                                   contract: FuturesContract) -> float:
    """Convert ETF-space stop distance to dollar risk per contract."""
    return stop_distance_etf * contract.multiplier
