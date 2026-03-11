#!/usr/bin/env bash
set -euo pipefail

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

usage() {
  cat <<'EOF'
Usage:
  bash run_auto.sh \
    --entry train_cls|train \
    --run-name <name> \
    [--mode ask|auto|resume|overwrite] \
    [--device cuda:0|cuda:1|cpu] \
    [--run-root <dir>] \
    [--project-dir <dir>] \
    [--input-dir <dataset_root>] \
    [--gpus <int>] \
    [--num-workers <int>] \
    [--epochs <int>] \
    [--batch-size <int>] \
    [--val-every <int>] \
    [--lr <float>] \
    [--pretrain-ckpt <path>] \
    [--extra-args "<raw args>"] \
    [--no-follow]
EOF
}

ENTRY=""
RUN_NAME=""
MODE="auto"
DEVICE="cuda:0"
GPUS=1
NUM_WORKERS=15
EPOCHS=""
BATCH_SIZE=""
VAL_EVERY=""
LR=""
PRETRAIN_CKPT=""
EXTRA_ARGS=""
FOLLOW_LOGS=1
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR"
RUN_ROOT="$PROJECT_DIR/runs"
INPUT_DIR="/workspace/source/datasets"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --entry) ENTRY="$2"; shift 2 ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --run-root) RUN_ROOT="$2"; shift 2 ;;
    --project-dir) PROJECT_DIR="$2"; shift 2 ;;
    --input-dir) INPUT_DIR="$2"; shift 2 ;;
    --gpus) GPUS="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --val-every) VAL_EVERY="$2"; shift 2 ;;
    --lr) LR="$2"; shift 2 ;;
    --pretrain-ckpt) PRETRAIN_CKPT="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --no-follow) FOLLOW_LOGS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

if [[ -z "$ENTRY" || -z "$RUN_NAME" ]]; then
  echo "Missing required args --entry and --run-name"
  usage
  exit 1
fi

if [[ "$ENTRY" != "train_cls" && "$ENTRY" != "train" ]]; then
  echo "--entry must be one of: train_cls, train"
  exit 1
fi

if [[ "$MODE" != "ask" && "$MODE" != "auto" && "$MODE" != "resume" && "$MODE" != "overwrite" ]]; then
  echo "--mode must be one of: ask, auto, resume, overwrite"
  exit 1
fi

RUN_DIR="$RUN_ROOT/$RUN_NAME"
CKPT_DIR="$RUN_DIR/checkpoints"
LOG_DIR="$RUN_DIR/logs"
META_DIR="$RUN_DIR/meta"
TRAIN_LOG="$LOG_DIR/train.log"
LAUNCH_LOG="$LOG_DIR/launch.log"
COMMAND_SNAPSHOT="$META_DIR/command.sh"
STATE_FILE="$META_DIR/run_state.json"
PID_FILE="$META_DIR/pid.txt"
RUNNER_FILE="$META_DIR/run_inner.sh"

mkdir -p "$CKPT_DIR" "$LOG_DIR" "$META_DIR"

log_launch() {
  echo "[$(timestamp)] $1" | tee -a "$LAUNCH_LOG"
}

find_resume_ckpt() {
  local candidate=""
  candidate="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*final*.ckpt' | sort | tail -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  candidate="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*last*.ckpt' | sort | tail -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  candidate="$(find "$CKPT_DIR" -maxdepth 1 -type f -name 'best*.ckpt' | sort | tail -n 1 || true)"
  if [[ -n "$candidate" ]]; then
    echo "$candidate"
    return
  fi
  candidate="$(find "$CKPT_DIR" -maxdepth 1 -type f -name '*.ckpt' | sort | tail -n 1 || true)"
  echo "$candidate"
}

EXISTING_CKPT="$(find_resume_ckpt)"
HAS_CKPT="false"
if [[ -n "$EXISTING_CKPT" ]]; then
  HAS_CKPT="true"
fi

if [[ "$MODE" == "ask" && "$HAS_CKPT" == "true" ]]; then
  echo "Checkpoint found: $EXISTING_CKPT"
  read -r -p "Resume or overwrite? [resume/overwrite]: " answer
  if [[ "$answer" == "resume" ]]; then
    MODE="resume"
  elif [[ "$answer" == "overwrite" ]]; then
    MODE="overwrite"
  else
    echo "Invalid choice: $answer"
    exit 1
  fi
fi

if [[ "$MODE" == "overwrite" && -d "$RUN_DIR" ]]; then
  rm -rf "$RUN_DIR"
  mkdir -p "$CKPT_DIR" "$LOG_DIR" "$META_DIR"
  log_launch "INFO removed previous run directory: $RUN_DIR"
fi

RESUME_CKPT=""
RUN_STATE="started"
if [[ "$MODE" == "resume" ]]; then
  if [[ "$HAS_CKPT" != "true" ]]; then
    log_launch "ERROR mode=resume but no checkpoint found in $CKPT_DIR"
    exit 1
  fi
  RESUME_CKPT="$EXISTING_CKPT"
  RUN_STATE="resumed"
elif [[ "$MODE" == "auto" ]]; then
  if [[ "$HAS_CKPT" == "true" ]]; then
    RESUME_CKPT="$EXISTING_CKPT"
    RUN_STATE="resumed"
  fi
elif [[ "$MODE" == "overwrite" ]]; then
  RUN_STATE="overwritten"
fi

CUDA_ENV=""
if [[ "$DEVICE" == cuda:* ]]; then
  CUDA_ENV="${DEVICE#cuda:}"
  export CUDA_VISIBLE_DEVICES="$CUDA_ENV"
  log_launch "INFO set CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES from device=$DEVICE"
else
  log_launch "INFO device=$DEVICE (no CUDA_VISIBLE_DEVICES export)"
fi

cmd=(python -u "$PROJECT_DIR/$ENTRY.py"
  --id checkpoints
  --logdir "$RUN_DIR"
  --input_dir "$INPUT_DIR"
  --gpus "$GPUS"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$EPOCHS" ]]; then cmd+=(--epochs "$EPOCHS"); fi
if [[ -n "$BATCH_SIZE" ]]; then cmd+=(--batch_size "$BATCH_SIZE"); fi
if [[ -n "$VAL_EVERY" ]]; then cmd+=(--val_every "$VAL_EVERY"); fi
if [[ -n "$LR" ]]; then cmd+=(--lr "$LR"); fi
if [[ -n "$RESUME_CKPT" ]]; then cmd+=(--resume_ckpt "$RESUME_CKPT"); fi
if [[ "$ENTRY" == "train" && -n "$PRETRAIN_CKPT" ]]; then cmd+=(--pretrain_ckpt "$PRETRAIN_CKPT"); fi

if [[ -n "$EXTRA_ARGS" ]]; then
  read -r -a extra_arr <<< "$EXTRA_ARGS"
  cmd+=("${extra_arr[@]}")
fi

runner_prefix=()
if command -v stdbuf >/dev/null 2>&1; then
  runner_prefix=(stdbuf -oL -eL)
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "cd %q\n" "$PROJECT_DIR"
  printf "%q " "${cmd[@]}"
  echo
} > "$COMMAND_SNAPSHOT"
chmod +x "$COMMAND_SNAPSHOT"

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "echo \$\$ > %q\n" "$PID_FILE"
  printf "cd %q\n" "$PROJECT_DIR"
  echo "set -o pipefail"
  printf "%q " "${runner_prefix[@]}" "${cmd[@]}"
  echo "2>&1 | awk '{ print strftime(\"[%Y-%m-%d %H:%M:%S] INFO\"), \$0; fflush(); }' | tee -a \"$TRAIN_LOG\""
} > "$RUNNER_FILE"
chmod +x "$RUNNER_FILE"

SESSION_NAME="train_${ENTRY}_${RUN_NAME}_$(date '+%m%d_%H%M%S')"

if command -v tmux >/dev/null 2>&1; then
  tmux new-session -d -s "$SESSION_NAME" "bash '$RUNNER_FILE'"
  sleep 1
  PANE_PID="$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_pid}' | head -n1 || true)"
  if [[ -z "$PANE_PID" ]]; then
    PANE_PID="unknown"
  fi
  if [[ ! -f "$PID_FILE" ]]; then
    echo "$PANE_PID" > "$PID_FILE"
  fi
  LAUNCHER="tmux"
else
  nohup bash "$RUNNER_FILE" >/dev/null 2>&1 &
  echo "$!" > "$PID_FILE"
  PANE_PID="$!"
  SESSION_NAME="nohup_${ENTRY}_${RUN_NAME}"
  LAUNCHER="nohup"
fi

PID_VALUE="$(cat "$PID_FILE" 2>/dev/null || echo "$PANE_PID")"

cat > "$STATE_FILE" <<EOF
{
  "state": "$RUN_STATE",
  "entry": "$ENTRY",
  "mode": "$MODE",
  "device": "$DEVICE",
  "launcher": "$LAUNCHER",
  "session_name": "$SESSION_NAME",
  "pid": "$PID_VALUE",
  "checkpoint_used": "$RESUME_CKPT",
  "project_dir": "$PROJECT_DIR",
  "run_root": "$RUN_ROOT",
  "run_name": "$RUN_NAME",
  "train_log": "$TRAIN_LOG",
  "updated_at": "$(timestamp)"
}
EOF

log_launch "INFO mode=$MODE resolved_state=$RUN_STATE entry=$ENTRY"
if [[ -n "$RESUME_CKPT" ]]; then
  log_launch "INFO resume checkpoint: $RESUME_CKPT"
fi
log_launch "INFO session name: $SESSION_NAME"
log_launch "INFO pid: $PID_VALUE"
log_launch "INFO train log: $TRAIN_LOG"

echo "session name: $SESSION_NAME"
echo "pid: $PID_VALUE"
echo "train log: $TRAIN_LOG"

if [[ "$FOLLOW_LOGS" -eq 1 ]]; then
  echo "live stream on this terminal (Ctrl+C to stop viewing; training still runs in background): $TRAIN_LOG"
  echo "background session: $SESSION_NAME ($LAUNCHER)"
  tail -n 50 -f "$TRAIN_LOG"
fi
