#!/bin/sh
# --serve binds loopback by default; --serve-any is the only way to
# listen on every interface, and it prints a warning. The engine binds
# before it opens the checkpoint, so no weights are needed: the test
# reads the listening socket from /proc/net/tcp while the engine waits
# for its one client, then stops it. Linux only (SKIP elsewhere);
# --serve itself is POSIX only.
# Usage: sh tests/test_serve_bind.sh [engine-binary]   (from littlenemo/)
set -u
BIN=${1:-bin/nemo-engine}
if [ ! -r /proc/net/tcp ]; then echo "SKIP: /proc/net/tcp not available (Linux only)"; exit 0; fi
if [ ! -x "$BIN" ]; then echo "SKIP: engine binary $BIN not built"; exit 0; fi
T=$(mktemp -d); trap 'rm -rf "$T"' EXIT
fail=0
hexport() { printf '%04X' "$1"; }
listen_addr() { # prints the local address (hex) listening on port $1, if any
    awk -v p=":$(hexport "$1")" '$4 == "0A" && substr($2, length($2) - 4) == p { print substr($2, 1, 8) }' /proc/net/tcp
}
run_case() { # name port expect_addr extra_flag
    name=$1; port=$2; want=$3; shift 3
    "$BIN" "$T/none" "$T/none.json" --serve "$port" "$@" 2> "$T/$name.err" &
    pid=$!
    got=""; i=0
    while [ $i -lt 50 ]; do got=$(listen_addr "$port"); [ -n "$got" ] && break; sleep 0.1; i=$((i + 1)); done
    kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null
    if [ "$got" = "$want" ]; then echo "ok   $name: listening address $got"; else echo "FAIL $name: wanted $want, got '${got:-none}'"; cat "$T/$name.err"; fail=1; fi
}
P1=$(( 20000 + $$ % 20000 )); P2=$(( P1 + 1 ))
run_case default "$P1" 0100007F
if grep -q "WARNING" "$T/default.err"; then echo "FAIL default: printed a warning"; fail=1; else echo "ok   default: no warning"; fi
run_case any "$P2" 00000000 --serve-any
if grep -q "WARNING: --serve-any" "$T/any.err"; then echo "ok   any: warning printed"; else echo "FAIL any: no warning"; fail=1; fi
"$BIN" "$T/none" "$T/none.json" --serve-any 2> "$T/lone.err"; rc=$?
if [ $rc -eq 2 ] && grep -q "needs --serve PORT" "$T/lone.err"; then echo "ok   --serve-any alone is refused"; else echo "FAIL --serve-any alone: rc=$rc"; fail=1; fi
if [ $fail -eq 0 ]; then echo "RESULT: ALL PASS"; else echo "RESULT: FAIL"; exit 1; fi
