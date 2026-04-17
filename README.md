# quant1

Paper-trading quant system running on Interactive Brokers (paper account) via a DigitalOcean VPS. Two codebases share a VPS and run seven strategies across five processes.

## Layout

```
quant1/
├── futures/     # MES, MNQ, MGC — ORB, VWAP reversion, overnight reversion
└── equities/    # SPY, QQQ, SMH, XLK, GLD/TLT — ORB, OpenDrive, pairs
```

## Strategies

| Process | Instruments | Strategy | clientId |
|---|---|---|---|
| trader-futures | MES, MNQ, MGC | ORB + VWAP Reversion (MGC VWAP only) | 1 |
| trader-overnight | MNQ | Overnight Reversion (8 PM – 2 AM ET) | 3 |
| trader-equity-orb | SPY, QQQ | Opening Range Breakout | 10 |
| trader-opendrive | SMH, XLK | OpenDrive | 11 |
| trader-pairs | GLD/TLT | Pairs spread | 12 |
| balance-monitor | — | 5-min health + equity to Discord | 20 |

## Setup

1. `cp .env.example .env` and fill in real values
2. `cp futures/scripts/vps_deploy/ibc_config.ini.example futures/scripts/vps_deploy/ibc_config.ini` and fill in IB credentials
3. `pip install -r futures/requirements.txt` (and equities/requirements.txt if present)

## Security

All secrets (VPS password, IB login, Discord webhooks) live in environment variables — **never** hardcoded. See `.env.example` for the full list.
