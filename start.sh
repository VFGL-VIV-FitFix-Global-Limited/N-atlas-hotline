#!/bin/sh
set -e
PORT="${PORT:-8000}"
echo "n-atlas-hotline binding 0.0.0.0:${PORT}"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --proxy-headers --forwarded-allow-ips '*'
