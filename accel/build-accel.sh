#!/bin/sh
# NTSE-o2 accel lanes (research, scratch) - POSIX, gcc or clang.
# Run from the REPO ROOT: sh accel/build-accel.sh
# Builds only under accel/ (accel/bin, accel/obj); no GPU SDK needed.
set -e
CC=${CC:-gcc}
FLAGS="-O2 -std=c99 -Wall -Wextra -D_FILE_OFFSET_BITS=64"
SRC="accel/src/accel_core.c accel/src/accel_cuda.c accel/src/accel_opencl.c"
LIBS="-lm -lpthread -ldl"
mkdir -p accel/bin accel/obj
for t in guard kernels stream real linear plan; do
    $CC $FLAGS -o accel/bin/test_accel_$t accel/tests/test_accel_$t.c $SRC $LIBS
done
$CC $FLAGS -o accel/bin/ntse-accel-bench accel/tools/ntse-accel-bench.c $SRC $LIBS
NFLAGS="$FLAGS -D_POSIX_C_SOURCE=200809L"
$CC $NFLAGS -mavx2 -c -o accel/obj/nn_simd.o littlenemo/src/nn_simd.c
for f in forward attn mamba loader json pool tokenizer; do
    $CC $NFLAGS -c -o accel/obj/$f.o littlenemo/src/$f.c
done
$CC $NFLAGS -Dnemo_linear_bf16=nemo_cpu_linear_bf16 \
    -Dnemo_linear_bf16_rows_auto=nemo_cpu_linear_bf16_rows_auto \
    -c -o accel/obj/nn_cpu.o littlenemo/src/nn.c
$CC $NFLAGS -o accel/bin/test_nemo_forward_accel \
    accel/littlenemo/test_nemo_forward_accel.c accel/littlenemo/nn_accel.c \
    accel/obj/forward.o accel/obj/attn.o accel/obj/mamba.o accel/obj/nn_cpu.o \
    accel/obj/nn_simd.o accel/obj/loader.o accel/obj/json.o accel/obj/pool.o \
    accel/obj/tokenizer.o $SRC $LIBS
echo "built accel/bin: 6 accel suites, ntse-accel-bench, test_nemo_forward_accel"
