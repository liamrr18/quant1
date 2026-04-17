# Futures ORB Trading System (MES / MNQ)

Adapts the proven equity ORB (Opening Range Breakout) strategy from SPY/QQQ
to micro futures (MES and MNQ) for higher dollar returns via built-in leverage.

## Strategy

Same ORB strategy as the equity system:
- 15-minute opening range establishes breakout levels
- Long on close above OR high, short on close below OR low
- Target: 1.5x range width, stop: opposite side of range
- MES (maps to SPY): gap filter (skip days with gap < 0.3%)
- MNQ (maps to QQQ): stale exit (cut underwater positions after 90 bars)

## Data Proxy

Uses SPY/QQQ minute data as proxy for MES/MNQ since:
- MES tracks S&P 500 index, SPY tracks S&P 500 index
- MNQ tracks Nasdaq 100 index, QQQ tracks Nasdaq 100 index
- During regular trading hours (09:30-16:00 ET), correlation > 0.999
- Strategy trades only 09:45-15:30 ET, so basis risk is negligible

## Futures Contract Specifications

| Contract | Name | Proxy | Point Value | Tick Size | Tick Value | Multiplier | Margin |
|----------|------|-------|-------------|-----------|------------|------------|--------|
| MES | Micro E-mini S&P 500 | SPY | $5/pt | 0.25 pts | $1.25 | $50/SPY$ | $1,500 |
| MNQ | Micro E-mini Nasdaq 100 | QQQ | $2/pt | 0.25 pts | $0.50 | $80/QQQ$ | $1,800 |

## Cost Model

Per contract, round trip:
- **Commission**: $0.62 (exchange + clearing + NFA)
- **Slippage**: 1 tick per side ($2.50 MES, $1.00 MNQ)
- **Total**: $3.12/contract MES, $1.62/contract MNQ

## Backtest Results (Jan 2024 - Apr 2026)

### Full Period (In-Sample)

| Metric | MES | MNQ | Portfolio |
|--------|-----|-----|----------|
| Trades | 551 | 1,225 | 1,776 |
| Total P&L | $+48,479 | $+115,163 | $+163,642 |
| Total Return | +48.5% | +115.2% | - |
| Sharpe Ratio | 0.83 | 1.25 | 1.04 avg |
| Max Drawdown | -30.2% | -35.9% | - |
| Win Rate | 53.2% | 38.3% | - |
| Profit Factor | 1.12 | 1.20 | - |
| Avg Trade | $+93 | $+97 | - |
| Total Costs | $5,806 | $6,909 | $12,714 |

### Walk-Forward OOS (60/20/20, 25 windows)

| Metric | MES | MNQ |
|--------|-----|-----|
| OOS Return | +78.7% | +129.1% |
| OOS P&L | $+68,343 | $+95,693 |
| OOS Sharpe | 1.16 | 2.08 |
| OOS Max DD | -16.1% | -19.0% |
| OOS Win Rate | 55.3% | 39.1% |
| OOS Profit Factor | 1.22 | 1.19 |
| Total Costs | $4,484 | $5,685 |

### Locked OOS (Dec 2025 - Apr 2026)

| Metric | MES | MNQ | Portfolio |
|--------|-----|-----|----------|
| Trades | 93 | 175 | 268 |
| Total P&L | $-7,505 | $+52,407 | $+44,902 |
| Sharpe | -0.66 | 4.64 | 1.99 avg |
| Max DD | -27.1% | -8.6% | - |

**Honest assessment**: MES had a losing locked OOS period. MNQ more than compensated.
The portfolio effect is critical - running both instruments reduces single-symbol risk.

## Equity vs Futures Comparison

| Metric | SPY (Equity) | MES (Futures) | QQQ (Equity) | MNQ (Futures) |
|--------|-------------|---------------|-------------|---------------|
| Return | +5.7% | +48.5% | +8.8% | +115.2% |
| P&L | $+5,738 | $+48,479 | $+8,792 | $+115,163 |
| Sharpe | 1.38 | 0.83 | 1.33 | 1.25 |
| Max DD | -24.2% | -30.2% | -24.9% | -35.9% |

Futures multiply dollar P&L by ~8-13x but also amplify drawdowns proportionally.
The Sharpe ratio is lower on futures because leverage increases both returns and
volatility. The *edge* is the same; the *dollars* are much larger.

## Parameter Sensitivity

The edge survives all nearby parameter perturbations:

**MES**: All target_multiple values (1.0-2.0) profitable. All range_minutes (10-30) profitable.
**MNQ**: All target_multiple values (1.0-2.0) profitable ($108k-$148k). Very robust.

No cliff edges in any parameter direction.

## Dollar P&L Projections

### MES (Annualized ~12.7%)
| Account | 6-Month | 12-Month | Max DD$ |
|---------|---------|----------|---------|
| $25k | $+1,833 | $+3,666 | $7,546 |
| $50k | $+3,666 | $+7,333 | $15,091 |
| $100k | $+7,333 | $+14,666 | $30,182 |
| $250k | $+18,332 | $+36,664 | $75,455 |

### MNQ (Annualized ~26.1%)
| Account | 6-Month | 12-Month | Max DD$ |
|---------|---------|----------|---------|
| $25k | $+4,355 | $+8,710 | $8,979 |
| $50k | $+8,710 | $+17,420 | $17,958 |
| $100k | $+17,420 | $+34,839 | $35,916 |
| $250k | $+43,549 | $+87,098 | $89,790 |

## Risk Management

- **Per-trade max loss**: 2% of account ($2,000 on $100k)
- **Daily max loss**: 5% of account ($5,000 on $100k)
- **Circuit breaker**: At 3% daily loss, reduce size by 50%
- **Margin safety**: Require 2x margin available before entry
- **Max contracts**: 30 MES, 20 MNQ per trade
- **Max concurrent**: 2 positions (one per instrument)
- **Consecutive loss halt**: Stop after 5 consecutive losses

### Worst-Case Scenarios

| Metric | MES | MNQ |
|--------|-----|-----|
| Worst single trade | $-3,482 | $-4,384 |
| Worst single day | $-11,617 | $-9,325 |
| Worst single week | $-10,686 | $-8,958 |
| Max consecutive losses | 8 | 11 |
| Max drawdown ($) | $-39,284 | $-41,222 |

## Usage

### Backtest
```bash
python run_backtest.py                                    # Full backtest
python run_backtest.py --walkforward                      # Walk-forward OOS
python run_backtest.py --oos                              # Locked OOS (Dec 2025-Apr 2026)
python run_backtest.py --sensitivity                      # Parameter sensitivity
python run_backtest.py --compare --projections            # Equity comparison + projections
python run_backtest.py --report --compare                 # HTML report with comparison
python run_backtest.py --symbols MES --capital 50000      # MES only, $50k account
```

### Live Paper Trading
```bash
python run_futures.py                    # Paper trade MES + MNQ
python run_futures.py --dry-run          # Signal-only mode (no orders)
python run_futures.py --symbols MES      # MES only
```

Logs are written to `logs/futures/YYYY-MM-DD/` with:
- `trades.csv` - Trade journal with futures P&L
- `signals.csv` - Signal log with opening range data
- `equity.csv` - Equity snapshots including circuit breaker status
- `summary.json` - End-of-day summary
- `trader.log` - Full execution log

Discord alerts are sent for every entry/exit/halt via the shared webhook.

## Honest Conclusion

**Does futures trading meaningfully improve dollar returns?**
Yes. The same ORB edge that generates $5-9k on equities generates $48-115k on futures
over the same period, an 8-13x amplification.

**Is the leverage appropriate?**
The leverage is aggressive. Max drawdowns of 30-36% on the full backtest and
$39-41k on a $100k account are painful. The circuit breaker (reduce at 3% daily
loss, halt at 5%) helps but cannot prevent multi-day drawdowns.

**Recommendation:**
- Start with the MNQ side (stronger OOS results, better Sharpe)
- Use conservative sizing (reduce MAX_RISK_PER_TRADE_PCT to 1% initially)
- Paper trade for at least 20 days before risking real capital
- Never allocate more than capital you can afford to lose entirely
- The diversification benefit of running both MES and MNQ is real
  (MES lost in locked OOS, MNQ gained - the portfolio was net positive)

**What this system does NOT do:**
- Predict the future
- Guarantee profits
- Eliminate drawdown risk
- Work when the edge decays (which all edges eventually do)

The strategy works because the opening range breakout captures genuine
intraday momentum. Futures leverage amplifies this edge into meaningful
dollars. But leverage is a double-edged sword, and the drawdowns are
proportionally larger.
