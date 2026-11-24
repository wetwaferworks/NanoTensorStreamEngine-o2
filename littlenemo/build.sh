#!/bin/sh
# Little Nemo build, Linux. Run from the littlenemo directory.
# Builds everything: the engine, the checkpoint inspector, and
# every test suite, into bin/.
set -e
mkdir -p bin
CC=${CC:-cc}
FLAGS="-O2 -std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L"
SRC="src/forward.c src/attn.c src/mamba.c src/nn.c src/loader.c src/json.c src/pool.c"
$CC $FLAGS -o bin/nemo-engine src/engine.c src/tokenizer.c $SRC -lm -pthread
$CC $FLAGS -o bin/nemo-inspect src/nemo.c src/tokenizer.c $SRC -lm -pthread
for t in test_mamba test_attn test_loader test_forward test_tokenizer \
         test_guards test_hostile test_coherence test_par; do
    $CC $FLAGS -o bin/$t tests/$t.c src/tokenizer.c $SRC -lm -pthread
done
echo "built bin/nemo-engine bin/nemo-inspect and 9 test suites in bin/"
