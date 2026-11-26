#!/bin/sh
# Regression test for the M-P8 remedy (Codex Astra M-P8,
# re-raised as ChatGPT Sol M8; remedy taken 2026-11-25):
# nq2-streamtest must treat an incomplete pack as a hard
# failure, never publish an accumulator for it.
#
# Background: with the old zero-fill read policy, the frozen
# fixture pack cut by 40 trailing bytes moved the accumulator
# from 31.102473211088157 to 138.46327352204358 and streamtest
# still exited 0. This test reproduces that cut (and a harsher
# one) against the frozen baseline pack and asserts, for the
# sequential path and one pipelined configuration:
#   - exit status is nonzero;
#   - stderr names the shortfall ("incomplete");
#   - no "accumulator:" line is printed.
#
# Usage: sh tools/test-streamtest-truncation.sh [streamtest-binary]
# Run from the repo root. The binary defaults to the scratch
# build if present, else ./nq2-streamtest from build.sh.
set -u
W=$(pwd)
BIN=${1:-$W/nq2-streamtest}
PACK=$W/results/sigma-full-b256-v2run.nq2
BOOK=$W/results/sigma-full-b256-v2run.nq2book
SHARDS="$W/sigma-3b/model-00001-of-00002.safetensors $W/sigma-3b/model-00002-of-00002.safetensors"
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

SIZE=$(wc -c < "$PACK")
fail=0
for CUT in 40 1000000; do
    head -c $((SIZE - CUT)) "$PACK" > "$T/trunc.nq2"
    for U in 0 2; do
        # shellcheck disable=SC2086
        "$BIN" --pack "$T/trunc.nq2" --book "$BOOK" \
            --shards $SHARDS --unpackers $U \
            > "$T/out.txt" 2> "$T/err.txt"
        rc=$?
        if [ $rc -eq 0 ]; then
            echo "FAIL cut=$CUT u=$U: exited 0 on a truncated pack"
            fail=1
        fi
        if ! grep -q "incomplete" "$T/err.txt"; then
            echo "FAIL cut=$CUT u=$U: stderr does not name the shortfall"
            cat "$T/err.txt"
            fail=1
        fi
        if grep -q "^accumulator:" "$T/out.txt"; then
            echo "FAIL cut=$CUT u=$U: an accumulator was published"
            fail=1
        fi
        echo "ok   cut=$CUT u=$U: rc=$rc, shortfall named, no accumulator"
    done
done
if [ $fail -eq 0 ]; then
    echo "RESULT: ALL PASS"
else
    echo "RESULT: FAIL"
    exit 1
fi
