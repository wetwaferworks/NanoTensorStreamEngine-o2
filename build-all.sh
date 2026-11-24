#!/bin/sh
# Nano Quant Engine, one-click build, Linux.
# Builds every tool: the six production tools and the fifteen
# experimental ones. Run from the repo root:
#
#     ./build-all.sh
#
# Everything is C99 with no dependencies beyond the standard
# library, libm, and pthreads.
set -e
./build.sh
./build-experiments.sh
echo "build-all: done (21 tools)"
