#!/bin/sh
# NanoQuatTwo research tools, Linux. Run from the repo root.
# These are the experiment benches from the 2026-11-23/24
# sessions (Gizmo rounds 2 and 3, normalization research). They
# are independently testable and none of them touches the
# production pipeline or the shipped baseline; build.sh builds
# the six production tools.
set -e
cc -O2 -std=c99 -Wall -Wextra -o nq2-tensorstat tools/nq2-tensorstat.c -lm
cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-streamtest tools/nq2-streamtest.c -lm
cc -O3 -march=native -std=c99 -Wall -Wextra -o nq2-rsqrtbench tools/nq2-rsqrtbench.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-hadamard tools/nq2-hadamard.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-pq tools/nq2-pq.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-additive tools/nq2-additive.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-framed tools/nq2-framed.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-fused tools/nq2-fused.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-entries tools/nq2-entries.c -lm
echo "built the nine research tools (see EXPERIMENTS-2026-11-24.md)"
