#!/usr/bin/env bash
# Restart the full dev stack: kills anything on the dev ports, then starts
# backend (8230) + web frontend (5173). Run from anywhere; Ctrl-C stops both.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BACKEND_PORT="${BACKEND_PORT:-8230}"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"

# Kill stale processes from a previous run
for port in "$BACKEND_PORT" "$FRONTEND_PORT"; do
    pids=$(lsof -ti tcp:"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "Killing stale process(es) on port $port: $pids"
        kill $pids 2>/dev/null || true
        sleep 0.5
    fi
done

cleanup() {
    echo
    echo "Stopping dev stack…"
    kill 0 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Starting backend on :$BACKEND_PORT"
(cd "$ROOT" && paperclaw serve --port "$BACKEND_PORT" --reload) &

echo "Starting web frontend on :$FRONTEND_PORT"
(cd "$ROOT/frontend" && npx vite --config vite.web.config.ts --port "$FRONTEND_PORT") &

wait
