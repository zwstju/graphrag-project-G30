#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/Projects/graphrag-project-G30}"
HERE="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$ROOT/src/backend"

cp "$HERE/src/backend/app_stage2.py" "$ROOT/src/backend/app_stage2.py"

rm -rf "$ROOT/frontend_stage2"
cp -r "$HERE/frontend_stage2" "$ROOT/frontend_stage2"

echo
echo "Installed into:"
echo "  $ROOT/src/backend/app_stage2.py"
echo "  $ROOT/frontend_stage2/"
echo
echo "Start with:"
echo "  cd $ROOT"
echo "  source .venv/bin/activate"
echo "  uvicorn src.backend.app_stage2:app --host 127.0.0.1 --port 8000"
echo
echo "Then open http://127.0.0.1:8000"
