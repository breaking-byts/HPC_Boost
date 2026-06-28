#!/usr/bin/env bash
set -euo pipefail

BINARY_ID="$1"
BINARY_PATH="$2"
ARGS="$3"
GROUP_ID="$4"
EVENTS="$5"
OUTDIR="$6"
CPU="${7:-1}"
TIMEOUT_SEC="${8:-30}"
INTERVAL_MS="${9:-10}"

mkdir -p "$OUTDIR"

META="$OUTDIR/metadata.json"
PERF="$OUTDIR/perf.csv"
STDOUT="$OUTDIR/stdout.txt"
STDERR="$OUTDIR/stderr.txt"

START_NS=$(date +%s%N)
set +e

taskset -c "$CPU" \
  perf stat -o "$PERF" -x, -I "$INTERVAL_MS" -e "{$EVENTS}" \
  -- timeout "${TIMEOUT_SEC}s" sh -c "$BINARY_PATH $ARGS" \
  >"$STDOUT" 2>"$STDERR"
STATUS=$?
set -e
END_NS=$(date +%s%N)

cat > "$META" <<JSON
{
  "binary_id": "$BINARY_ID",
  "binary_path": "$BINARY_PATH",
  "args": "$ARGS",
  "group_id": "$GROUP_ID",
  "events": "$EVENTS",
  "cpu": "$CPU",
  "timeout_sec": "$TIMEOUT_SEC",
  "interval_ms": "$INTERVAL_MS",
  "exit_status": "$STATUS",
  "start_ns": "$START_NS",
  "end_ns": "$END_NS",
  "kernel": "$(uname -r)",
  "hostname": "$(hostname)"
}
JSON

exit 0
