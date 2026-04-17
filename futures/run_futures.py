#!/usr/bin/env python3
"""Futures paper trading launcher via Interactive Brokers.

Connects to IB TWS or Gateway and trades MES/MNQ micro futures using
the ORB strategy. Requires TWS or IB Gateway running with API enabled.

Usage:
    python run_futures.py                # Paper trading (live)
    python run_futures.py --dry-run      # Signals only, no orders
    python run_futures.py --symbols MES  # MES only
    python run_futures.py --port 4002    # Use IB Gateway port

IMPORTANT: This does NOT modify or interfere with the equity trading
system. It uses separate log directories (logs/futures/) and trades
actual futures contracts on IB.
"""

import argparse
import logging
import sys
import os
from datetime import datetime

import pytz

from trading.config import (
    FUTURES_SYMBOLS, ORB_SHARED_DEFAULTS,
    SYMBOL_PROFILES, VWAP_REVERSION_DEFAULTS, LOG_DIR,
    IB_HOST, IB_PORT, IB_CLIENT_ID,
)
from trading.strategies.orb import ORBBreakout
from trading.strategies.vwap_reversion import VWAPReversion
from trading.data.contracts import CONTRACTS
from trading.live.trader import FuturesLiveTrader

ET = pytz.timezone("America/New_York")


def make_orb_for_symbol(futures_symbol: str) -> ORBBreakout:
    """Create ORB strategy with per-symbol profile."""
    params = dict(ORB_SHARED_DEFAULTS)
    if futures_symbol in SYMBOL_PROFILES:
        params.update(SYMBOL_PROFILES[futures_symbol])
    return ORBBreakout(**params)


def make_vwap_for_symbol(futures_symbol: str) -> VWAPReversion:
    """Create VWAP Reversion strategy."""
    return VWAPReversion(**VWAP_REVERSION_DEFAULTS)


def verify_ib_connection(port: int = None) -> dict:
    """Verify IB connection and return account info.

    Returns account info dict or raises with clear instructions.
    """
    from trading.execution.broker import get_ib, get_account_equity, get_available_margin, get_contract_month

    try:
        ib = get_ib()
    except RuntimeError as e:
        print("\n" + "=" * 60)
        print("  ERROR: Cannot connect to Interactive Brokers")
        print("=" * 60)
        print(f"\n  Connection failed at {IB_HOST}:{port or IB_PORT}")
        print()
        print("  To fix this, you need IB TWS or IB Gateway running:")
        print()
        print("  OPTION 1: IB Trader Workstation (TWS)")
        print("    1. Download from: https://www.interactivebrokers.com/en/trading/tws.php")
        print("    2. Install and launch TWS")
        print("    3. Log in with your IB paper trading credentials")
        print("    4. Go to: File > Global Configuration > API > Settings")
        print("    5. Check 'Enable ActiveX and Socket Clients'")
        print("    6. Set Socket port to 7497 (paper trading)")
        print("    7. Uncheck 'Read-Only API'")
        print("    8. Click Apply/OK")
        print(f"    9. Re-run: python run_futures.py --port 7497")
        print()
        print("  OPTION 2: IB Gateway (lighter, no GUI)")
        print("    1. Download from: https://www.interactivebrokers.com/en/trading/ibgateway-stable.php")
        print("    2. Install and launch IB Gateway")
        print("    3. Log in with paper trading credentials")
        print("    4. Select 'IB API' as connection type")
        print("    5. The default paper port is 4002")
        print(f"    6. Re-run: python run_futures.py --port 4002")
        print()
        print("  Paper trading account:")
        print("    - If you don't have one, enable it at:")
        print("      https://www.interactivebrokers.com/en/trading/paperTrading.php")
        print("    - Paper credentials are separate from live credentials")
        print("=" * 60)
        sys.exit(1)

    equity = get_account_equity()
    margin = get_available_margin()

    # Resolve contracts to verify they work
    contract_months = {}
    for sym in FUTURES_SYMBOLS:
        try:
            month = get_contract_month(sym)
            contract_months[sym] = month
        except Exception as e:
            contract_months[sym] = f"ERROR: {e}"

    return {
        "equity": equity,
        "margin": margin,
        "contracts": contract_months,
    }


def main():
    parser = argparse.ArgumentParser(description="Futures Paper Trading (IB)")
    parser.add_argument("--symbols", nargs="+", default=FUTURES_SYMBOLS,
                        help="Futures symbols (default: MES MNQ)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Signal scanning only, no orders")
    parser.add_argument("--port", type=int, default=None,
                        help="IB port (7497=TWS paper, 4002=Gateway paper)")
    parser.add_argument("--client-id", type=int, default=None,
                        help="IB client ID (default from config)")
    args = parser.parse_args()

    # Override IB settings from command line
    if args.port is not None:
        import trading.config as cfg
        cfg.IB_PORT = args.port
        # Also update the broker module's reference
        import trading.execution.broker as broker_mod
        # The broker reads from config at call time via get_ib()

    if args.client_id is not None:
        import trading.config as cfg
        cfg.IB_CLIENT_ID = args.client_id

    # Setup logging
    os.makedirs(LOG_DIR, exist_ok=True)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    log_dir = os.path.join(LOG_DIR, today)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "trader.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file),
        ],
    )
    log = logging.getLogger(__name__)

    # Verify IB connection before starting
    print(f"\nConnecting to IB at {IB_HOST}:{args.port or IB_PORT}...")
    acct = verify_ib_connection(args.port)

    # Validate symbols and build strategies
    # ORB runs on MES/MNQ only. VWAP runs on MES/MNQ/MGC.
    orb_symbols = [s for s in args.symbols if s in SYMBOL_PROFILES]
    strategies = {}
    vwap_strategies = {}
    for sym in args.symbols:
        if sym not in CONTRACTS:
            log.error("Unknown futures symbol: %s (supported: %s)",
                      sym, list(CONTRACTS.keys()))
            sys.exit(1)
        if sym in orb_symbols:
            strategies[sym] = make_orb_for_symbol(sym)
        vwap_strategies[sym] = make_vwap_for_symbol(sym)

    # Print startup info
    mode = "DRY RUN" if args.dry_run else "PAPER TRADING"
    print(f"\n{'='*60}")
    print(f"  FUTURES MULTI-STRATEGY TRADER - {mode} (Interactive Brokers)")
    print(f"{'='*60}")
    print(f"  Date:             {today}")
    print(f"  Log file:         {log_file}")
    print(f"  IB connection:    {IB_HOST}:{args.port or IB_PORT} (clientId={args.client_id or IB_CLIENT_ID})")
    print(f"  Account equity:   ${acct['equity']:,.2f}")
    print(f"  Available margin: ${acct['margin']:,.2f}")
    print()
    for sym in args.symbols:
        contract = CONTRACTS[sym]
        vwap_strat = vwap_strategies[sym]
        month = acct["contracts"].get(sym, "unknown")
        print(f"  {sym} ({contract.name}):")
        print(f"    Contract:      {month}")
        print(f"    Multiplier:    ${contract.multiplier}/point")
        print(f"    Margin:        ${contract.margin_intraday:,.0f}/contract")
        if sym in strategies:
            print(f"    ORB params:    {strategies[sym].get_params()}")
        print(f"    VWAP params:   {vwap_strat.get_params()}")
    print(f"{'='*60}\n")

    # Launch trader (ORB is primary; VWAP Reversion runs alongside)
    trader = FuturesLiveTrader(
        strategies=strategies,
        vwap_strategies=vwap_strategies,
        symbols=args.symbols,
        dry_run=args.dry_run,
    )
    trader.run()


if __name__ == "__main__":
    main()
