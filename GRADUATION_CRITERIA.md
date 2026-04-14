# Candidate Strategy Graduation Criteria

A candidate strategy must pass ALL four gates before being promoted
to the production portfolio. No exceptions.

## Gate 1: Code Correctness

- [ ] Strategy module produces identical backtest results to research (within 5%)
- [ ] Strategy works in both backtest engine and live trader
- [ ] Logging produces correct CSV files (trades, signals, equity)
- [ ] Strategy resets properly on new trading day
- [ ] No lookahead bias in signal generation

## Gate 2: Paper Trading Alignment (minimum 20 trading days)

- [ ] Trade count within 2x of backtest daily average
- [ ] Win rate within 10 percentage points of backtest
- [ ] Average trade PnL direction matches backtest (positive if backtest positive)
- [ ] No API error rate above 1%
- [ ] No interference with ORB baseline positions
- [ ] No slippage exceeding 3x the backtest assumption ($0.03/share)
- [ ] Equity curve behaves reasonably (no spikes, no unexplained jumps)

## Gate 3: Robustness Checks

All 7 validation dimensions must pass:

- [ ] **Walk-forward sensitivity**: positive Sharpe on all 3 window configs (40/20/20, 60/20/20, 80/20/20)
- [ ] **Parameter sensitivity**: >50% of ±20% perturbations maintain positive OOS Sharpe
- [ ] **Subperiod analysis**: ≥3 of 4 dev quarters profitable
- [ ] **Slippage stress**: Sharpe > 0 at 2x base slippage ($0.02/share)
- [ ] **Trade distribution**: max consecutive losers < 10, no single trade loss > 1% of equity
- [ ] **Alpha/beta**: alpha positive on both dev and OOS periods
- [ ] **Regime analysis**: neither high-vol nor low-vol regime produces negative total return

## Gate 4: Portfolio Value-Add

- [ ] Adding strategy improves locked OOS portfolio Sharpe
- [ ] Correlation with ORB baseline < 0.30
- [ ] Does not breach global position limit (MAX_CONCURRENT_POSITIONS = 4)
- [ ] Combined daily loss across all strategies does not exceed 3% of equity
- [ ] No single-instrument concentration above 40% of portfolio equity

## Automatic Fail Conditions

Any of these immediately disqualifies a candidate:

- Paper Sharpe < 0 after 30 trading days
- Paper max drawdown > 3x backtest max drawdown
- Any single-day loss > 2% of equity
- Trades outside market hours
- Strategy produces signals inconsistent with backtest logic
- Unresolved API errors or position tracking discrepancies

## Decision Process

1. Strategy must pass Gate 1 before paper trading begins
2. Paper trading runs for minimum 20 trading days
3. After 20 days: evaluate Gates 2 and 3
4. If Gates 1-3 pass: evaluate Gate 4
5. If ALL gates pass: strategy is APPROVED for production portfolio
6. If any gate fails: strategy is SUSPENDED pending investigation

## Current Candidates

| Strategy | Gate 1 | Gate 2 | Gate 3 | Gate 4 | Status |
|----------|--------|--------|--------|--------|--------|
| Pairs GLD/TLT | PASS (parity verified) | PENDING (paper trading) | PENDING | PENDING | Paper trading |
| OpenDrive SMH | PASS (parity verified) | PENDING (paper trading) | PENDING | PENDING | Paper trading |
| OpenDrive XLK | PASS (parity verified) | PENDING (paper trading) | PENDING | PENDING | Paper trading |
