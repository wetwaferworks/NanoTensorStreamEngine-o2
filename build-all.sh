#!/bin/sh
# Nano Quant Engine, one-click build, Linux.
# Builds every tool plus Little Nemo. Run from the repo root:
#
#     ./build-all.sh
#
# build.sh already builds all 23 tools and Little Nemo, so this
# script is just the one-click name for it. (It used to run
# build-experiments.sh afterwards as well, which recompiled the
# research tools a second time for no benefit... Archie review
# P2-1, 2026-11-24. build-experiments.sh remains as the
# research-subset build for working on those tools alone.)
#
# Everything is C99 with no dependencies beyond the standard
# library, libm, and pthreads.
set -e
./build.sh
echo "build-all: done (23 tools + Little Nemo)"
