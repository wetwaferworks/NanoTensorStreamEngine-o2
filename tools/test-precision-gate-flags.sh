#!/bin/sh
# CLI checks for the Precision Gate flags in nq2-infer-triforce.
# Does not load a pack or run the model. Pass the engine path
# as $1, or leave it to find build/lanes or the current directory.
set -eu
if [ "${1:-}" != "" ]; then
    ENG=$1
elif [ -x ./build/lanes/nq2-infer-triforce ]; then
    ENG=./build/lanes/nq2-infer-triforce
elif [ -x ./build/lanes/nq2-infer-triforce.exe ]; then
    ENG=./build/lanes/nq2-infer-triforce.exe
elif [ -x ./nq2-infer-triforce ]; then
    ENG=./nq2-infer-triforce
elif [ -x ./nq2-infer-triforce.exe ]; then
    ENG=./nq2-infer-triforce.exe
else
    echo "usage: $0 PATH-TO-nq2-infer-triforce" >&2
    exit 2
fi

fails=0
ok() { echo "ok   $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }

# Usage text names the postures and the gate flags.
out=$($ENG 2>&1) || true
echo "$out" | grep -q -- '--posture evidence|auto|fast|think-harder' && ok "usage lists postures" || fail "usage lists postures"
echo "$out" | grep -q -- '--gate-lookahead' && ok "usage lists --gate-lookahead" || fail "usage lists --gate-lookahead"
echo "$out" | grep -q -- '--gate-resync' && ok "usage lists --gate-resync" || fail "usage lists --gate-resync"
echo "$out" | grep -q -- '--gate-eps' && ok "usage lists --gate-eps" || fail "usage lists --gate-eps"
echo "$out" | grep -q -- '--gate-budget' && ok "usage lists --gate-budget" || fail "usage lists --gate-budget"
echo "$out" | grep -q -- '--gate-log' && ok "usage lists --gate-log" || fail "usage lists --gate-log"

# Named refusals, no model work.
out=$($ENG --posture not-a-posture --dir x --ids 1 2>&1) || true
echo "$out" | grep -q "is not a posture" && ok "unknown posture is a named refusal" || fail "unknown posture is a named refusal"

out=$($ENG --posture auto --gate-lookahead 9 --dir x --ids 1 2>&1) || true
echo "$out" | grep -q "lookahead must be in \[1, 4\]" && ok "lookahead 9 is refused" || fail "lookahead 9 is refused"

out=$($ENG --gate-resync -1 --dir x --ids 1 2>&1) || true
echo "$out" | grep -q "resync must be >= 0" && ok "resync -1 is refused" || fail "resync -1 is refused"

out=$($ENG --gate-eps 0 --dir x --ids 1 2>&1) || true
echo "$out" | grep -q "gate-eps" && ok "non-positive --gate-eps is refused" || fail "non-positive --gate-eps is refused"

if [ "$fails" -eq 0 ]; then
    echo "RESULT: ALL PASS"
    exit 0
fi
echo "RESULT: FAILURES $fails"
exit 1
