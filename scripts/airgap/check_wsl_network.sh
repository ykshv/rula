#!/usr/bin/env bash
set -euo pipefail

TARGET_URL="${1:-https://example.com}"

if command -v curl >/dev/null 2>&1; then
  if curl --connect-timeout 3 --max-time 5 -fsS "$TARGET_URL" >/tmp/ru-local-avatar-airgap.out 2>/tmp/ru-local-avatar-airgap.err; then
    echo "WSL/container outbound network is reachable. Air-gap gate failed." >&2
    exit 1
  fi
  echo "WSL/container outbound network check blocked as expected."
  exit 0
fi

echo "curl is required for WSL air-gap check." >&2
exit 2
