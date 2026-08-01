#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

if ! command -v node >/dev/null 2>&1; then
  echo "Node.js is required. Install Node.js 22.12.0 or newer."
  exit 1
fi

NODE_VERSION="$(node --version | sed 's/^v//')"
IFS=. read -r NODE_MAJOR NODE_MINOR _ <<< "$NODE_VERSION"

if (( NODE_MAJOR < 22 || (NODE_MAJOR == 22 && NODE_MINOR < 12) )); then
  echo "Node.js $NODE_VERSION is not supported. Install Node.js 22.12.0 or newer."
  exit 1
fi

if [ ! -d node_modules ]; then
  echo "Installing dependencies..."
  npm install
fi

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-${1:-4321}}"

echo "Starting Soopace at http://$HOST:$PORT/"
exec npm run dev -- --host "$HOST" --port "$PORT"
