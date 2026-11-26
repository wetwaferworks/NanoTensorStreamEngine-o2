#!/bin/sh
# Build the scaler test binary and a shared library (libnemo_scale.so /
# nemo_scale.dll). Uses only the compiler already on PATH. No installs.
set -e
cd "$(dirname "$0")"
CC="${CC:-gcc}"
CFLAGS="-O2 -std=c11 -Wall -Wextra -Werror -fPIC -DNEMO_SCALE_BUILD"
# test binary
$CC $CFLAGS -o test_nemo_scale test_nemo_scale.c nemo_scale.c -lm
# shared library
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*|Windows_NT)
    $CC $CFLAGS -shared -o nemo_scale.dll nemo_scale.c -lm
    ;;
  *)
    $CC $CFLAGS -shared -o libnemo_scale.so nemo_scale.c -lm
    ;;
esac
echo "built in $(pwd)"
ls -l test_nemo_scale libnemo_scale.so nemo_scale.dll 2>/dev/null || true
