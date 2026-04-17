#!/bin/bash
# Create systemd services for IB Gateway + all 5 traders
set -e

mkdir -p /var/log/ibc /var/log/traders /root/futures_trader/logs/futures /root/flamboyant-lewin/logs/opendrive /root/flamboyant-lewin/logs/pairs

# ─── IB Gateway service ───────────────────────────────────────────────
cat > /etc/systemd/system/ibgateway.service <<'EOF'
[Unit]
Description=IB Gateway (via IBC + Xvfb)
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/opt/ibc
ExecStartPre=/bin/bash -c "pkill -9 Xvfb || true; pkill -9 java || true; sleep 2"
ExecStartPre=/bin/bash -c "Xvfb :1 -screen 0 1280x1024x24 &"
ExecStartPre=/bin/sleep 3
ExecStart=/opt/ibc/scripts/ibcstart.sh 1045 -g --tws-path=/root/Jts --ibc-path=/opt/ibc --ibc-ini=/opt/ibc/config.ini --mode=paper
Environment=DISPLAY=:1
Restart=always
RestartSec=60
StandardOutput=append:/var/log/ibgateway.log
StandardError=append:/var/log/ibgateway.log

[Install]
WantedBy=multi-user.target
EOF

# ─── Futures ORB+VWAP (clientId=1) ────────────────────────────────────
cat > /etc/systemd/system/trader-futures.service <<'EOF'
[Unit]
Description=Futures ORB+VWAP Trader
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/futures_trader
Environment=PATH=/opt/trader_env/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 30
ExecStart=/opt/trader_env/bin/python -u run_futures.py
Restart=always
RestartSec=60
StandardOutput=append:/var/log/traders/futures.log
StandardError=append:/var/log/traders/futures.log

[Install]
WantedBy=multi-user.target
EOF

# ─── Overnight (clientId=3) ───────────────────────────────────────────
cat > /etc/systemd/system/trader-overnight.service <<'EOF'
[Unit]
Description=Overnight Reversion Trader
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/futures_trader
Environment=PATH=/opt/trader_env/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 35
ExecStart=/opt/trader_env/bin/python -u run_overnight.py
Restart=always
RestartSec=60
StandardOutput=append:/var/log/traders/overnight.log
StandardError=append:/var/log/traders/overnight.log

[Install]
WantedBy=multi-user.target
EOF

# ─── Equity ORB (clientId=10) ─────────────────────────────────────────
cat > /etc/systemd/system/trader-equity-orb.service <<'EOF'
[Unit]
Description=Equity ORB Trader (SPY+QQQ)
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/flamboyant-lewin
Environment=PATH=/opt/trader_env/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 40
ExecStart=/opt/trader_env/bin/python -u run_live.py
Restart=always
RestartSec=60
StandardOutput=append:/var/log/traders/equity-orb.log
StandardError=append:/var/log/traders/equity-orb.log

[Install]
WantedBy=multi-user.target
EOF

# ─── OpenDrive (clientId=11) ──────────────────────────────────────────
cat > /etc/systemd/system/trader-opendrive.service <<'EOF'
[Unit]
Description=OpenDrive Trader (SMH+XLK)
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/flamboyant-lewin
Environment=PATH=/opt/trader_env/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 45
ExecStart=/opt/trader_env/bin/python -u run_opendrive.py
Restart=always
RestartSec=60
StandardOutput=append:/var/log/traders/opendrive.log
StandardError=append:/var/log/traders/opendrive.log

[Install]
WantedBy=multi-user.target
EOF

# ─── Pairs (clientId=12) ──────────────────────────────────────────────
cat > /etc/systemd/system/trader-pairs.service <<'EOF'
[Unit]
Description=Pairs Trader (GLD/TLT)
After=ibgateway.service
Requires=ibgateway.service

[Service]
Type=simple
User=root
WorkingDirectory=/root/flamboyant-lewin
Environment=PATH=/opt/trader_env/bin:/usr/bin:/bin
ExecStartPre=/bin/sleep 50
ExecStart=/opt/trader_env/bin/python -u run_pairs.py
Restart=always
RestartSec=60
StandardOutput=append:/var/log/traders/pairs.log
StandardError=append:/var/log/traders/pairs.log

[Install]
WantedBy=multi-user.target
EOF

# Reload and enable all
systemctl daemon-reload
systemctl enable ibgateway.service trader-futures.service trader-overnight.service trader-equity-orb.service trader-opendrive.service trader-pairs.service

echo "All systemd services created and enabled."
systemctl list-unit-files | grep -E "ibgateway|trader-"
