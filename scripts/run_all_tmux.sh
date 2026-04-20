#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION_NAME="adaptive_tracking_run"
CONFIG_PATH="configs/default.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

cd "$PROJECT_ROOT"

CURRENT_TMUX_SESSION=""
if [[ -n "${TMUX:-}" ]]; then
  CURRENT_TMUX_SESSION="$(tmux display-message -p '#S' 2>/dev/null || true)"
fi

if [[ "$CURRENT_TMUX_SESSION" != "$SESSION_NAME" ]]; then
  if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "tmux session '$SESSION_NAME' already exists" >&2
    exit 1
  fi

  mkdir -p "$PROJECT_ROOT/outputs/logs"
  TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
  LOG_PATH="$PROJECT_ROOT/outputs/logs/run_all_${TIMESTAMP}.log"
  STATUS_PATH="$PROJECT_ROOT/outputs/logs/latest_progress.json"

  tmux new-session -d -s "$SESSION_NAME" \
    "cd '$PROJECT_ROOT' && ADAPTIVE_TRACKING_LOG_PATH='$LOG_PATH' ADAPTIVE_TRACKING_STATUS_PATH='$STATUS_PATH' bash -lc './scripts/run_all_tmux.sh --config $CONFIG_PATH'"

  echo "session=$SESSION_NAME"
  echo "log=$LOG_PATH"
  echo "status=$STATUS_PATH"
  echo "attach=tmux attach -t $SESSION_NAME"
  echo "progress=cat $STATUS_PATH"
  exit 0
fi

LOG_PATH="${ADAPTIVE_TRACKING_LOG_PATH:-$PROJECT_ROOT/outputs/logs/run_all_$(date +%Y%m%d_%H%M%S).log}"
STATUS_PATH="${ADAPTIVE_TRACKING_STATUS_PATH:-$PROJECT_ROOT/outputs/logs/latest_progress.json}"
mkdir -p "$(dirname "$LOG_PATH")"

log_bootstrap() {
  local stage_progress="$1"
  local overall_progress="$2"
  local message="$3"
  printf '%s | INFO | bootstrap | stage=bootstrap stage_progress=%.1f overall_progress=%.1f message=%s\n' \
    "$(date '+%Y-%m-%d %H:%M:%S')" "$stage_progress" "$overall_progress" "$message" >> "$LOG_PATH"
  python3 - "$STATUS_PATH" "$stage_progress" "$overall_progress" "$message" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

status_path = Path(sys.argv[1]).resolve()
status_path.parent.mkdir(parents=True, exist_ok=True)
payload = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "stage": "bootstrap",
    "stage_progress": float(sys.argv[2]),
    "overall_progress": float(sys.argv[3]),
    "message": sys.argv[4],
}
status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY
}

log_bootstrap 0.0 0.0 "starting environment bootstrap"
python3 -m venv "$PROJECT_ROOT/.venv" >> "$LOG_PATH" 2>&1
log_bootstrap 40.0 2.0 "virtualenv ready"
"$PROJECT_ROOT/.venv/bin/python" -m pip install --upgrade pip >> "$LOG_PATH" 2>&1
"$PROJECT_ROOT/.venv/bin/python" -m pip install -r "$PROJECT_ROOT/requirements.txt" >> "$LOG_PATH" 2>&1
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/check_cuda_env.py" --json >> "$LOG_PATH" 2>&1 || true
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/install_torch.py" --mode auto --require-cuda-if-available >> "$LOG_PATH" 2>&1
"$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/check_cuda_env.py" --json >> "$LOG_PATH" 2>&1
log_bootstrap 100.0 5.0 "environment ready"

exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/run_all_with_progress.py" \
  --config "$CONFIG_PATH" \
  --log-path "$LOG_PATH" \
  --status-path "$STATUS_PATH"
