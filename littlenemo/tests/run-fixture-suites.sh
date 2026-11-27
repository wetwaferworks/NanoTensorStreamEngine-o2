#!/bin/sh
# Runs the Little Nemo suites that need crafted or partial checkpoints:
# test_hostile, test_coherence and test_prefill.
#
#   sh tests/run-fixture-suites.sh <checkpoint_dir> [fixture_dir]
#
# Run from the littlenemo directory after build.sh. The fixtures are made
# by bin/mk_fixtures under fixture_dir (default /tmp/nemo-fixtures): tiny
# crafted files, one sparse file with no model data, and links to the first
# two shards of the checkpoint. Keep fixture_dir outside the repository.
[ -n "$1" ] || { echo "usage: run-fixture-suites.sh <checkpoint_dir> [fixture_dir]"; exit 2; }
CKPT=$1
FX=${2:-/tmp/nemo-fixtures}
failed=0

bin/mk_fixtures "$CKPT" "$FX" || { echo "fixture build failed"; exit 1; }

for d in "$FX"/hostile/refuse-*; do
    if bin/test_hostile refuse "$d" >/dev/null; then
        echo "ok   hostile $(basename "$d")"
    else
        echo "FAIL hostile $(basename "$d")"; failed=1
    fi
done
bin/test_hostile i32 "$FX/hostile/i32" || failed=1
bin/test_coherence "$FX/partial" "$FX/dtype" || failed=1
bin/test_prefill "$CKPT" || failed=1

if [ "$failed" = 1 ]; then echo "fixture suites: FAILURES"; exit 1; fi
echo "fixture suites: ALL PASS"
