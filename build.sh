#!/bin/sh
# NanoQuatTwo build, Linux. Run from the repo root.
# Builds everything: all 23 tools in tools/ and the Little Nemo
# engine with its test suites (littlenemo/bin/).
set -e
CC=${CC:-cc}
FLAGS="-O2 -std=c99 -Wall -Wextra -D_FILE_OFFSET_BITS=64"
for t in quantize run audit frameprobe errprobe tensorstat entries \
         hadamard pq additive framed fused correct rpq \
         dc dc-adv logitcmp; do
    $CC $FLAGS -o nq2-$t tools/nq2-$t.c -lm
done
# rsqrtbench is a benchmark: its AVX2/AVX-512 sections are guarded
# by __AVX2__/__AVX512F__ and silently vanish without the flags
# (Archie P2-3). Build it like build-experiments.sh does.
$CC -O3 -march=native -std=c99 -Wall -Wextra -o nq2-rsqrtbench tools/nq2-rsqrtbench.c -lm
for t in infer infer-lab infer-triforce infer-triforce-dbg streamtest; do
    $CC $FLAGS -pthread -o nq2-$t tools/nq2-$t.c -lm
done
(cd littlenemo && ./build.sh)
echo "built all 23 nq2 tools and Little Nemo (littlenemo/bin/)"
