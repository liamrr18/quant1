#!/bin/bash
# Start IB Gateway unattended via IBC + Xvfb
set -e

# Kill any leftovers
pkill -9 Xvfb 2>/dev/null || true
pkill -9 java 2>/dev/null || true
sleep 2

# Start virtual display
Xvfb :1 -screen 0 1280x1024x24 &
export DISPLAY=:1
sleep 3

# Find IB Gateway's actual version directory (e.g. "1037")
IBGW_VERSION=$(ls /opt/ibgateway | grep -oE 'IB Gateway [0-9]+\.[0-9]+' | head -1 | grep -oE '[0-9]+' | head -1)
if [ -z "$IBGW_VERSION" ]; then
    # Try standard paths
    if [ -d /opt/ibgateway ]; then
        IBGW_VERSION="1037"  # fallback
    fi
fi

# Paths
export TWS_MAJOR_VRSN="${IBGW_VERSION:-1037}"
export IBC_INI=/opt/ibc/config.ini
export TRADING_MODE=paper
export IBC_PATH=/opt/ibc
export TWS_PATH=/opt/ibgateway
export TWS_SETTINGS_PATH=/root/Jts
export LOG_PATH=/var/log/ibc
export JAVA_PATH=
export TWOFA_TIMEOUT_ACTION=exit

mkdir -p "$LOG_PATH" "$TWS_SETTINGS_PATH"

# Launch IBC wrapper
exec /opt/ibc/scripts/ibcstart.sh \
    "${TWS_MAJOR_VRSN}" \
    -g \
    --tws-path="${TWS_PATH}" \
    --tws-settings-path="${TWS_SETTINGS_PATH}" \
    --ibc-path="${IBC_PATH}" \
    --ibc-ini="${IBC_INI}" \
    --mode="${TRADING_MODE}"
