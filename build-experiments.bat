@echo off
REM NanoQuatTwo research tools, Windows (MSYS2/MinGW). Run from
REM the repo root. Experiment benches only; build.bat builds
REM the six production tools. See EXPERIMENTS-2026-11-24.md.
setlocal
gcc -O2 -std=c99 -Wall -Wextra -o nq2-tensorstat.exe tools\nq2-tensorstat.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -pthread -o nq2-streamtest.exe tools\nq2-streamtest.c -lm || goto :err
gcc -O3 -march=native -std=c99 -Wall -Wextra -o nq2-rsqrtbench.exe tools\nq2-rsqrtbench.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-hadamard.exe tools\nq2-hadamard.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-pq.exe tools\nq2-pq.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-additive.exe tools\nq2-additive.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-framed.exe tools\nq2-framed.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-fused.exe tools\nq2-fused.c -lm || goto :err
gcc -O2 -std=c99 -Wall -Wextra -o nq2-entries.exe tools\nq2-entries.c -lm || goto :err
echo built the nine research tools
exit /b 0
:err
echo BUILD FAILED
exit /b 1
