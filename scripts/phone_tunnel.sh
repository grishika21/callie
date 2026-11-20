#!/usr/bin/env bash
# The phone line behind a Cloudflare quick tunnel, with the Twilio number pointed at it.
#
#   scripts/phone_tunnel.sh
#
# 1. starts `cloudflared tunnel --url http://127.0.0.1:8770` (only the phone gateway is exposed;
#    the staff web app and the Support Desk stay on 127.0.0.1), and waits for its https URL;
# 2. points the Twilio number at it (python -m callie.phone.setup; prints console steps if the
#    Twilio settings aren't in .env);
# 3. runs phone_server.py with PHONE_PUBLIC_URL set. Ctrl-C stops both.
# A quick tunnel's URL changes on every run; this script re-points the number each time.
set -euo pipefail
cd "$(dirname "$0")/.."
CLOUDFLARED="${CLOUDFLARED:-$(command -v cloudflared || echo "$HOME/.local/bin/cloudflared")}"
PORT="${PHONE_PORT:-8770}"
LOG="$(mktemp -t phone-tunnel.XXXXXX.log)"

# http2 (TCP 443) by default: QUIC (UDP 7844) is often blocked on office and home networks.
"$CLOUDFLARED" tunnel --no-autoupdate --protocol "${CLOUDFLARED_PROTOCOL:-http2}" --url "http://127.0.0.1:${PORT}" \
  >"$LOG" 2>&1 &
TUNNEL=$!
trap 'kill $TUNNEL 2>/dev/null; rm -f "$LOG"' EXIT

URL=""
for _ in $(seq 1 60); do
  URL="$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG" | head -1 || true)"
  [[ -n "$URL" ]] && break
  sleep 1
done
[[ -n "$URL" ]] || { echo "cloudflared didn't report a URL; its log:" >&2; cat "$LOG" >&2; exit 1; }
for _ in $(seq 1 60); do  # the URL exists before the connection does
  grep -q "Registered tunnel connection" "$LOG" && break
  sleep 1
done
grep -q "Registered tunnel connection" "$LOG" || { echo "the tunnel didn't connect; its log:" >&2; cat "$LOG" >&2; exit 1; }
echo "tunnel: $URL -> http://127.0.0.1:${PORT}"

.venv/bin/python -m callie.phone.setup --url "$URL"
PHONE_PUBLIC_URL="$URL" .venv/bin/python phone_server.py
