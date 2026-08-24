#!/usr/bin/env bash
#
# Backend (FastAPI, :8000) ve frontend'i (Vite, :5173) tek komutla ayaga
# kaldirir. Ctrl+C ikisini birden durdurur.
#
# Kullanim:
#   ./execute.sh                # ikisini de baslat
#   ./execute.sh --backend-only # sadece backend
#   ./execute.sh --frontend-only # sadece frontend
#   ./execute.sh --help

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_PY="$ROOT_DIR/venv/bin/python"
UVICORN="$ROOT_DIR/venv/bin/uvicorn"
BACKEND_PORT=8000
FRONTEND_PORT=5173

MODE="all"
case "${1:-}" in
  --backend-only) MODE="backend" ;;
  --frontend-only) MODE="frontend" ;;
  --help|-h)
    sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
    ;;
  "") ;;
  *)
    echo "Bilinmeyen secenek: $1  (--help ile kullanim)" >&2
    exit 1
    ;;
esac

# --- on kosul kontrolleri ----------------------------------------------------

if [[ "$MODE" != "frontend" ]]; then
  if [[ ! -x "$UVICORN" ]]; then
    echo "HATA: venv bulunamadi ($UVICORN)." >&2
    echo "Kurulum icin README.md 'Kurulum' bolumune bakin:" >&2
    echo "  python3 -m venv venv && ./venv/bin/pip install -r requirements.txt" >&2
    exit 1
  fi
  if [[ ! -f "$ROOT_DIR/model/govde/adapter_model.safetensors" ]] \
      || [[ ! -f "$ROOT_DIR/model/basliklar.pt" ]]; then
    echo "HATA: egitilmis model yok (model/govde/ + model/basliklar.pt)." >&2
    echo "Once calistirin: ./venv/bin/python -m src.train" >&2
    exit 1
  fi
fi

if [[ "$MODE" != "backend" ]]; then
  if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
    echo "HATA: frontend bagimliliklari kurulu degil." >&2
    echo "Once calistirin: npm install --prefix frontend" >&2
    exit 1
  fi
fi

# Portlar zaten kullanimdaysa erken ve net bir hata ver.
port_dolu() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -i ":$1" -sTCP:LISTEN >/dev/null 2>&1
  else
    (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null && exec 3>&- 2>/dev/null
  fi
}

if [[ "$MODE" != "frontend" ]] && port_dolu "$BACKEND_PORT"; then
  echo "HATA: $BACKEND_PORT portu zaten kullanimda. Baska bir backend mi calisiyor?" >&2
  exit 1
fi
if [[ "$MODE" != "backend" ]] && port_dolu "$FRONTEND_PORT"; then
  echo "HATA: $FRONTEND_PORT portu zaten kullanimda. Baska bir frontend mi calisiyor?" >&2
  exit 1
fi

# --- baslatma -----------------------------------------------------------------

PIDS=()

cleanup() {
  echo
  echo "Kapatiliyor..."
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

if [[ "$MODE" != "frontend" ]]; then
  echo "-> backend baslatiliyor (http://127.0.0.1:$BACKEND_PORT) ..."
  "$UVICORN" backend.main:app --host 127.0.0.1 --port "$BACKEND_PORT" &
  PIDS+=($!)

  # Model yuklemesi birkac saniye surer; /health 200 doene kadar bekle.
  for _ in $(seq 1 60); do
    if curl -s -m 1 "http://127.0.0.1:$BACKEND_PORT/health" >/dev/null 2>&1; then
      echo "   backend hazir."
      break
    fi
    sleep 1
  done
fi

if [[ "$MODE" != "backend" ]]; then
  echo "-> frontend baslatiliyor (http://localhost:$FRONTEND_PORT) ..."
  npm run dev --prefix frontend -- --port "$FRONTEND_PORT" &
  PIDS+=($!)
fi

echo
echo "Calisiyor. Durdurmak icin Ctrl+C."
wait
