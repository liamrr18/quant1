# Intraday Quantitative Trading System

Automated intraday trading system running multiple strategy families on Alpaca. Trades liquid ETFs with strict risk management, walk-forward validation, and honest out-of-sample testing.

## Strategies

### Production (SPY+QQQ ORB)

Opening Range Breakout on SPY and QQQ. Trades the first 15-minute range breakout with per-symbol profiles, volume confirmation, ATR filtering, and gap filters.

- **Locked OOS Sharpe**: 3.35
- **Alpha**: +6.08% annualized, Beta: -0.036
- **Max Drawdown**: -0.55%
- **Trades**: ~3/day

### Candidates (paper trading)

Three new strategies are running in parallel paper trading, pending 20-day forward validation before any production promotion.

**OpenDrive XLK** (Opening Drive Continuation on XLK)
- Trades in the direction of the first 5-minute move after open
- Locked OOS Sharpe: 3.26, Alpha: +7.4%
- Robustness: 7/7 validation checks passed

**OpenDrive SMH** (Opening Drive Continuation on SMH)
- Same strategy family, different instrument and target multiple
- Locked OOS Sharpe: 3.87, Alpha: +17.1%
- Robustness: 6/7 (fails regime analysis)

**Pairs GLD/TLT** (Pairs Spread Mean Reversion)
- Trades log-spread mean reversion between GLD and TLT
- Locked OOS Sharpe: 4.86, Alpha: +15.1%
- Robustness: 6/7 (fails subperiod consistency)
- Caveat: dev Sharpe was only 0.49 — OOS may be regime-favorable

### Combined Portfolio (backtest, risk-parity weighted)

| Config | Locked OOS Sharpe | Return | MaxDD | Alpha |
|--------|:-:|:-:|:-:|:-:|
| ORB baseline only | 3.35 | +2.17% | -0.55% | +6.1% |
| All 5 streams (risk-parity) | 6.88 | +3.82% | -0.16% | +10.6% |
| All 5 streams (28-month full) | 2.58 | +7.52% | -0.81% | +3.5% |

Risk-parity weights: ORB SPY 22.9%, ORB QQQ 15.4%, Pairs 30.2%, OD SMH 10.5%, OD XLK 21.0%

## Setup

### Requirements

- Python 3.12+
- Alpaca paper trading account

### Install

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
ALPACA_PAPER=true
```

## Usage

### Paper Trading

```bash
# Production baseline (SPY+QQQ ORB)
python run_live.py

# OpenDrive candidates (SMH+XLK)
python run_opendrive.py

# Pairs candidate (GLD/TLT)
python run_pairs.py

# Dry run (signals only, no orders)
python run_live.py --dry-run
python run_opendrive.py --dry-run
python run_pairs.py --dry-run

# Single symbol
python run_opendrive.py --symbols XLK
```

### Backtesting

```bash
# Run backtest with per-symbol ORB profiles
python run_backtest.py

# Walk-forward validation
python run_backtest.py --walkforward

# Specific symbols and date range
python run_backtest.py --symbols SPY QQQ --start 2025-01-02 --end 2026-04-04
```

### Monitoring

```bash
# Daily report (run after market close, or anytime)
python daily_report.py
python daily_report.py 2026-04-15

# Weekly HTML dashboard (auto-opens in browser)
python weekly_report.py

# Forward validation (paper vs backtest comparison)
python forward_validation.py

# End-of-day alert (auto-opens Notepad — runs via Task Scheduler at 4:05 PM)
python eod_alert.py
```

### Research

```bash
# Robustness validation (7-dimension check)
python research/robustness_validation.py

# Backtest parity check (verify code matches research)
python research/parity_check.py

# Extended historical backtest (Jan 2024 - Apr 2026)
python research/extended_backtest.py
```

## Project Structure

```
run_live.py             ORB SPY+QQQ paper trading launcher
run_opendrive.py        OpenDrive SMH+XLK paper trading launcher
run_pairs.py            Pairs GLD/TLT paper trading launcher
run_backtest.py         Backtesting CLI
daily_report.py         Daily P&L summary across all strategies
weekly_report.py        Weekly HTML dashboard (paper vs backtest)
forward_validation.py   Paper trading alignment checker
eod_alert.py            End-of-day Notepad alert
setup_scheduler.bat     Windows Task Scheduler setup (run as admin)

trading/
  strategies/           Signal generation (ORB, OpenDrive, PairsSpread)
  backtest/             Backtesting engine and walk-forward validation
  live/                 Live trading (LiveTrader, PairsLiveTrader)
  execution/            Alpaca order execution (limit orders + market fallback)
  data/                 Market data fetching and feature engineering
  risk/                 Position sizing, daily loss limits, circuit breakers
  reporting/            HTML dashboard generation
  discord_alerts.py     Discord webhook trade notifications
  alerts.py             Email alert module (optional)

research/
  EXPERIMENT_LEDGER.md  Full decision log (19 experiments)
  robustness_validation.py   7-dimension validation suite
  parity_check.py            Backtest parity verification
  extended_backtest.py       Extended history analysis
  archive/                   Completed research scripts (29 files)

logs/                   ORB paper trading logs (per-day)
logs/opendrive/         OpenDrive paper trading logs
logs/pairs/             Pairs paper trading logs
data/cache/             Cached minute-bar data (Jan 2024 - Apr 2026)
```

## Methodology

- **Validation**: Walk-forward (60-day train, 20-day test, 20-day step) on dev period, locked OOS holdout
- **Data**: Jan 2024-Apr 2026 (28 months, extended); strategies tuned on Jan-Nov 2025 only
- **Data splits**: Dev Jan-Nov 2025, Locked OOS Dec 2025-Apr 2026
- **Cost model**: $0.01/share slippage, $0.00 commission (Alpaca)
- **Position sizing**: 30% of equity per trade (20% for candidates)
- **Risk controls**: 2% daily loss limit, 5 consecutive loss halt, 4 max concurrent positions
- **Anti-overfitting**: never iterate on OOS results, full metrics at every step, reject dev improvements that hurt OOS

## Logs

Each trader writes daily logs to its own directory:

| Trader | Log directory |
|--------|:--|
| ORB baseline | `logs/YYYY-MM-DD/` |
| OpenDrive | `logs/opendrive/YYYY-MM-DD/` |
| Pairs | `logs/pairs/YYYY-MM-DD/` |

Each directory contains:
- `trades.csv` — every entry/exit with P&L
- `signals.csv` — every signal scan with action taken
- `equity.csv` — minute-by-minute equity snapshots
- `summary.json` — end-of-day summary
- `trader.log` — full execution log

## Candidate Promotion

New strategies must pass all 4 gates in [GRADUATION_CRITERIA.md](GRADUATION_CRITERIA.md) before joining the production portfolio:

1. **Code Correctness** — backtest parity verified
2. **Paper Trading Alignment** — 20+ days, behavior matches research
3. **Robustness Checks** — 7 validation dimensions (walk-forward sensitivity, parameter perturbation, subperiod analysis, slippage stress, trade distribution, alpha/beta, regime analysis)
4. **Portfolio Value-Add** — improves combined Sharpe, correlation < 0.30
