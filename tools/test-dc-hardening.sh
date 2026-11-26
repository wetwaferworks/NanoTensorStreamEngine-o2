#!/bin/sh
# Regression test for the DTC code review P1-4 remedies
# (CODE-REVIEW-DTC-2026-11-25): nq2-dc must REFUSE malformed
# inputs with a nonzero exit and a named reason, never read
# them as weights silently and never publish NaN metrics
# under a success exit.
#
# Cases, all against the shipped Sigma pack/book/shards:
#   1. a book with one NaN value patched in at offset 12
#      (the first prototype float) must be refused with
#      "not finite" on stderr;
#   2. a crafted shard whose tensor span contradicts its
#      shape (F32 [2,2] = 16 bytes, span declares 8) must
#      be refused with "reject" on stderr;
#   3. a crafted shard whose span matches its shape but
#      runs past the data section (span 16, file carries 8
#      data bytes) must be refused the same way;
#   4. control: the unmodified book passes the --checksum
#      path with the recorded expansion checksum MATCH.
#
# Usage: sh tools/test-dc-hardening.sh [dc-binary]
# Run from the nanoquattwo root. The binary defaults to
# ./nq2-dc (build.sh output in this tree).
set -u
W=$(pwd)
BIN=${1:-$W/nq2-dc}
PACK=$W/results/sigma-full-b256-v2run.nq2
BOOK=$W/results/sigma-full-b256-v2run.nq2book
SHARD1=$W/sigma-3b/model-00001-of-00002.safetensors
SHARD2=$W/sigma-3b/model-00002-of-00002.safetensors
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail=0

# --- case 1: NaN book -------------------------------------
cp "$BOOK" "$T/nan.nq2book"
# fp32 quiet NaN, little-endian bytes 00 00 c0 7f, at the
# first book float (book header is 12 bytes).
printf '\000\000\300\177' | dd of="$T/nan.nq2book" bs=1 seek=12 \
    count=4 conv=notrunc 2>/dev/null
"$BIN" "$PACK" "$T/nan.nq2book" "$SHARD1" "$SHARD2" \
    > "$T/out1.txt" 2> "$T/err1.txt"
rc=$?
if [ $rc -eq 0 ]; then
    echo "FAIL nan-book: exited 0 on a NaN book value"
    fail=1
fi
if ! grep -q "not finite" "$T/err1.txt"; then
    echo "FAIL nan-book: stderr does not name the refusal"
    cat "$T/err1.txt"
    fail=1
fi
echo "ok   nan-book: rc=$rc, refusal named"

# --- cases 2/3: crafted shards ----------------------------
# Minimal safetensors: 8-byte header length, JSON header,
# then a short data section. make_shard SPAN DATA_BYTES.
make_shard() {
    JSON="{\"t\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,$1]}}"
    LEN=$(printf '%s' "$JSON" | wc -c)
    OUT=$2
    head -c 8 /dev/zero > "$OUT"
    printf '%s' "$JSON" >> "$OUT"
    head -c "$3" /dev/zero >> "$OUT"
    printf "\\$(printf '%03o' "$LEN")" | dd of="$OUT" bs=1 seek=0 \
        count=1 conv=notrunc 2>/dev/null
}
make_shard 8 "$T/badspan.safetensors" 64
make_shard 16 "$T/pastend.safetensors" 8
for NAME in badspan pastend; do
    "$BIN" "$PACK" "$BOOK" "$T/$NAME.safetensors" \
        > "$T/out-$NAME.txt" 2> "$T/err-$NAME.txt"
    rc=$?
    if [ $rc -eq 0 ]; then
        echo "FAIL $NAME: exited 0 on an invalid span"
        fail=1
    fi
    if ! grep -q "reject" "$T/err-$NAME.txt"; then
        echo "FAIL $NAME: stderr does not name the refusal"
        cat "$T/err-$NAME.txt"
        fail=1
    fi
    echo "ok   $NAME: rc=$rc, refusal named"
done

# --- case 4: control --------------------------------------
"$BIN" "$PACK" "$BOOK" --checksum \
    > "$T/outc.txt" 2> "$T/errc.txt"
rc=$?
if [ $rc -ne 0 ]; then
    echo "FAIL control: --checksum exited $rc on valid inputs"
    cat "$T/errc.txt"
    fail=1
fi
if ! grep -q "MATCH" "$T/outc.txt"; then
    echo "FAIL control: expansion checksum did not MATCH"
    cat "$T/outc.txt"
    fail=1
fi
echo "ok   control: rc=$rc, checksum MATCH"

if [ $fail -eq 0 ]; then
    echo "RESULT: ALL PASS"
else
    echo "RESULT: FAIL"
    exit 1
fi
