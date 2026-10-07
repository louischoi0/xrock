#!/usr/bin/env bash
# Project-hosted KDS server: ./kds_server with conf/kds.conf, state under var/kds/.
#   scripts/kds.sh start | stop | restart | status | log
# stop is graceful (a STOP statement flushes pages); a kill is a crash for KDS, so it is only the last resort.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BIN="$ROOT/kds_server"
CONF="$ROOT/conf/kds.conf"
STATE="$ROOT/var/kds"
PIDFILE="$STATE/kds.pid"
XROCK="${XROCK_BIN:-$ROOT/build/xrock}"
XCONF="${XROCK_CONF:-$ROOT/conf/xrock.conf}"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
ping() { "$XROCK" db ping -c "$XCONF" >/dev/null 2>&1; }

start() {
  if running; then echo "kds: already running (pid $(cat "$PIDFILE"))"; return 0; fi
  [[ -x "$BIN" ]] || { echo "kds: $BIN missing or not executable" >&2; exit 1; }
  [[ -x "$XROCK" ]] || { echo "kds: $XROCK missing - build xrock first" >&2; exit 1; }
  mkdir -p "$STATE"
  nohup "$BIN" --config "$CONF" >>"$STATE/server.out" 2>&1 &
  echo $! >"$PIDFILE"
  for _ in $(seq 50); do
    if ping; then echo "kds: started (pid $(cat "$PIDFILE"))"; return 0; fi
    running || break
    sleep 0.2
  done
  echo "kds: failed to start; tail of $STATE/server.out:" >&2
  tail -5 "$STATE/server.out" >&2
  rm -f "$PIDFILE"
  exit 1
}

stop() {
  if ! running; then echo "kds: not running"; rm -f "$PIDFILE"; return 0; fi
  local pid; pid="$(cat "$PIDFILE")"
  "$XROCK" db stop -c "$XCONF" >/dev/null || true
  for _ in $(seq 100); do
    kill -0 "$pid" 2>/dev/null || { rm -f "$PIDFILE"; echo "kds: stopped"; return 0; }
    sleep 0.2
  done
  echo "kds: pid $pid did not exit after STOP; not killing it (a kill is a crash). Check $STATE/kdb.log" >&2
  exit 1
}

status() {
  if running && ping; then echo "kds: running (pid $(cat "$PIDFILE")), answering";
  elif running; then echo "kds: process $(cat "$PIDFILE") alive but not answering"; exit 1;
  else echo "kds: not running"; exit 3; fi
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  log) tail -n "${2:-50}" "$STATE/kdb.log" ;;
  *) echo "usage: $0 start|stop|restart|status|log [n]" >&2; exit 2 ;;
esac
