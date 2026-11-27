#!/bin/sh
# Little Nemo build, Linux. Run from the littlenemo directory.
# Builds everything: the engine, the checkpoint inspector, and
# every test suite, into bin/.
set -e
mkdir -p bin obj
CC=${CC:-cc}
FLAGS="-O2 -std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64"
# nn_simd.c alone gets AVX2 codegen on x86-64 (H9): it holds the
# AVX2 kernel, reached only through a runtime CPUID check and
# only when the engine is told --simd. Everything else, nn.c's
# scalar kernel and dispatcher included, builds at the baseline
# instruction set so the fallback path is safe on pre-AVX
# hardware.
SIMD=""
case "$(uname -m)" in x86_64|amd64) SIMD="-mavx2" ;; esac
$CC $FLAGS $SIMD -c -o obj/nn_simd.o src/nn_simd.c
for f in forward attn mamba loader json pool tokenizer nn; do
    $CC $FLAGS -c -o obj/$f.o src/$f.c
done
LIB="obj/forward.o obj/attn.o obj/mamba.o obj/nn.o obj/nn_simd.o obj/loader.o obj/json.o obj/pool.o obj/tokenizer.o"
$CC $FLAGS -o bin/nemo-engine src/engine.c $LIB -lm -pthread
$CC $FLAGS -o bin/nemo-inspect src/nemo.c $LIB -lm -pthread
for t in test_mamba test_attn test_loader test_forward test_tokenizer \
         test_tokenizer_split test_guards test_hostile test_coherence \
         test_par test_simd test_prefill mk_fixtures; do
    $CC $FLAGS -o bin/$t tests/$t.c $LIB -lm -pthread
done
echo "built bin/nemo-engine bin/nemo-inspect and 12 test suites plus the fixture builder mk_fixtures in bin/"
