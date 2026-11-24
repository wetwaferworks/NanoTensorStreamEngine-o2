#!/bin/sh
# NanoQuantTwo experimental tools, Linux. Run from the repo root.
# The experiment benches and experimental engines from the
# 2026-11-23/24 sessions. None of them touches the shipped
# baseline; build.sh builds the six production tools, and
# build-all.sh builds everything in one go.
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
cc -O2 -std=c99 -Wall -Wextra -o nq2-correct tools/nq2-correct.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-rpq tools/nq2-rpq.c -lm
cc -O2 -std=c99 -Wall -Wextra -o nq2-logitcmp tools/nq2-logitcmp.c -lm
cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer-lab tools/nq2-infer-lab.c -lm
cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer-triforce tools/nq2-infer-triforce.c -lm
cc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-infer-triforce-dbg tools/nq2-infer-triforce-dbg.c -lm
echo "built the fifteen experimental tools (see EXPERIMENTS-2026-11-24.md and PIPELINE-2026-11-24.md)"
