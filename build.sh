#!/bin/sh
# NanoQuatTwo build, Linux. Run from the repo root.
# Builds all six tools: the quantizer, the streaming runner,
# the auditor, the inference engine, and the two probes.
set -e
cc -O2 -std=c99 -Wall -Wextra -o nq2-quantize tools/nq2-quantize.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-run tools/nq2-run.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-audit tools/nq2-audit.c -lm
cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer tools/nq2-infer.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-frameprobe tools/nq2-frameprobe.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-errprobe tools/nq2-errprobe.c -lm
echo "built ./nq2-quantize ./nq2-run ./nq2-audit ./nq2-infer ./nq2-frameprobe ./nq2-errprobe"
