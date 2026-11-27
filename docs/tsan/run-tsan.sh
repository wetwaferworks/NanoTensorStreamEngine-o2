#!/usr/bin/env bash
# ThreadSanitizer runs for the zero-mutex engines and tests.
# Requires gcc with libtsan (set CC to clang if you have it).
# Run from the repository root, on Linux or WSL:
#   ./docs/tsan/run-tsan.sh
# Binaries and raw logs land in build/lanes/tsan/ (gitignored).
set -euo pipefail
# Ubuntu 24.04 and WSL default vm.mmap_rnd_bits=32 maps into
# TSan reserved address space ("unexpected memory mapping").
# setarch -R disables ASLR for this process tree only.
if command -v setarch >/dev/null 2>&1 && [ -z "${TSAN_ADDR_NORAND:-}" ]; then
    export TSAN_ADDR_NORAND=1
    SCRIPT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)/$(basename "$0")
    exec setarch "$(uname -m)" -R /usr/bin/env bash "$SCRIPT" "$@"
fi
ROOT=$(CDPATH= cd -- "$(dirname "$0")/../.." && pwd)
cd "$ROOT"
OUT=${TSAN_OUT:-$ROOT/build/lanes/tsan}
mkdir -p "$OUT/bin" "$OUT/obj" "$OUT/logs"

CC=${CC:-gcc}
# -O1 is the usual TSan recommendation: enough optimisation to
# be a real engine build, enough debug info to name a report.
CFLAGS="-O1 -g -std=c99 -Wall -Wextra -pthread -fsanitize=thread -D_FILE_OFFSET_BITS=64"
export TSAN_OPTIONS="${TSAN_OPTIONS:-halt_on_error=1:abort_on_error=0:report_bugs=1:exitcode=66}"

fails=0
ok() { echo "ok   $1"; }
fail() { echo "FAIL $1"; fails=$((fails + 1)); }

echo "compiler: $($CC --version | head -1)"
echo "tsan-options: $TSAN_OPTIONS"
echo "out: $OUT"

# ---- handshake pool (1, 2 and 12 workers; hashes must match) ----
$CC $CFLAGS -I tools -o "$OUT/bin/test-handshake" tools/test-handshake.c
"$OUT/bin/test-handshake" 1 64 | tee "$OUT/logs/handshake-1.log"
"$OUT/bin/test-handshake" 2 64 | tee "$OUT/logs/handshake-2.log"
"$OUT/bin/test-handshake" 12 64 | tee "$OUT/logs/handshake-12x64.log"
# Same worker count, more rounds: stress only. The last hash
# changes with the round count, so it is not compared above.
"$OUT/bin/test-handshake" 12 256 | tee "$OUT/logs/handshake-12.log"
hash1=$(awk '/^hash /{print $2}' "$OUT/logs/handshake-1.log")
hash2=$(awk '/^hash /{print $2}' "$OUT/logs/handshake-2.log")
hash12=$(awk '/^hash /{print $2}' "$OUT/logs/handshake-12x64.log")
grep -q "RESULT: .* 0 failures" "$OUT/logs/handshake-1.log" && ok "handshake 1 worker" || fail "handshake 1 worker"
grep -q "RESULT: .* 0 failures" "$OUT/logs/handshake-2.log" && ok "handshake 2 workers" || fail "handshake 2 workers"
grep -q "RESULT: .* 0 failures" "$OUT/logs/handshake-12x64.log" && ok "handshake 12 workers (64 rounds)" || fail "handshake 12 workers (64 rounds)"
grep -q "RESULT: .* 0 failures" "$OUT/logs/handshake-12.log" && ok "handshake 12 workers (256 rounds)" || fail "handshake 12 workers (256 rounds)"
if [ -n "$hash1" ] && [ "$hash1" = "$hash2" ] && [ "$hash2" = "$hash12" ]; then
    ok "handshake hashes identical across 1/2/12 workers ($hash12)"
else
    fail "handshake hashes identical (1=$hash1 2=$hash2 12=$hash12)"
fi

# ---- DTC adversarial (single translation unit, no model) ----
$CC $CFLAGS -I tools -o "$OUT/bin/nq2-dc-adv" tools/nq2-dc-adv.c -lm
if "$OUT/bin/nq2-dc-adv" > "$OUT/logs/nq2-dc-adv.log" 2>&1; then
    ok "nq2-dc-adv under TSan (exit 0)"
else
    fail "nq2-dc-adv under TSan"
    tail -20 "$OUT/logs/nq2-dc-adv.log" || true
fi

# ---- Little Nemo unit suites that do not need a private checkpoint ----
LFLAGS="$CFLAGS -D_POSIX_C_SOURCE=200809L"
SIMD=""
case "$(uname -m)" in x86_64|amd64) SIMD="-mavx2" ;; esac
$CC $LFLAGS $SIMD -c -o "$OUT/obj/nn_simd.o" littlenemo/src/nn_simd.c
for f in forward attn mamba loader json pool tokenizer nn; do
    $CC $LFLAGS -c -o "$OUT/obj/$f.o" littlenemo/src/$f.c
done
LIB="$OUT/obj/forward.o $OUT/obj/attn.o $OUT/obj/mamba.o $OUT/obj/nn.o $OUT/obj/nn_simd.o $OUT/obj/loader.o $OUT/obj/json.o $OUT/obj/pool.o $OUT/obj/tokenizer.o"
for t in test_mamba test_attn test_guards test_simd; do
    $CC $LFLAGS -o "$OUT/bin/$t" littlenemo/tests/$t.c $LIB -lm -pthread
    if "$OUT/bin/$t" > "$OUT/logs/$t.log" 2>&1; then
        ok "littlenemo $t under TSan"
    else
        fail "littlenemo $t under TSan"
        tail -20 "$OUT/logs/$t.log" || true
    fi
done
$CC $LFLAGS -o "$OUT/bin/test_tokenizer" littlenemo/tests/test_tokenizer.c $LIB -lm -pthread
if "$OUT/bin/test_tokenizer" littlenemo/tests/fixture_tokenizer.json \
        > "$OUT/logs/test_tokenizer.log" 2>&1; then
    ok "littlenemo test_tokenizer under TSan"
else
    fail "littlenemo test_tokenizer under TSan"
    tail -20 "$OUT/logs/test_tokenizer.log" || true
fi

# Optional checkpoint suites stay off unless the caller points at
# a local tree: they are not required for the committed receipt.
if [ "${NEMO_CKPT:-}" != "" ] && [ -d "$NEMO_CKPT" ]; then
    for t in test_loader test_forward test_prefill test_par; do
        $CC $LFLAGS -o "$OUT/bin/$t" littlenemo/tests/$t.c $LIB -lm -pthread
    done
    if "$OUT/bin/test_par" "$NEMO_CKPT" 12 8 > "$OUT/logs/test_par.log" 2>&1; then
        ok "littlenemo test_par under TSan"
    else
        fail "littlenemo test_par under TSan"
        tail -20 "$OUT/logs/test_par.log" || true
    fi
else
    echo "note: NEMO_CKPT unset; skipped test_par / checkpoint suites"
fi

echo "RESULT: $fails failures"
exit $fails
