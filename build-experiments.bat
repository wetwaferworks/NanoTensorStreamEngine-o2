@echo off
rem NanoQuantTwo experimental tools, Windows with MinGW gcc.
rem Run from the repo root. Statically linked. The inference
rem engines (nq2-infer-lab, nq2-infer-triforce) need a
rem POSIX-thread MinGW (winpthreads); the probes build with
rem either thread model. build.bat builds the six production
rem tools; build-all.bat builds everything in one go.
setlocal
set CFLAGS=-O2 -std=c99 -Wall -Wextra -static
gcc %CFLAGS% -o nq2-tensorstat.exe tools\nq2-tensorstat.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -pthread -o nq2-streamtest.exe tools\nq2-streamtest.c
if %errorlevel% neq 0 goto fail
gcc -O3 -std=c99 -Wall -Wextra -static -o nq2-rsqrtbench.exe tools\nq2-rsqrtbench.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-hadamard.exe tools\nq2-hadamard.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-pq.exe tools\nq2-pq.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-additive.exe tools\nq2-additive.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-framed.exe tools\nq2-framed.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-fused.exe tools\nq2-fused.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-entries.exe tools\nq2-entries.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-correct.exe tools\nq2-correct.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-rpq.exe tools\nq2-rpq.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -o nq2-logitcmp.exe tools\nq2-logitcmp.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -pthread -o nq2-infer-lab.exe tools\nq2-infer-lab.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -pthread -o nq2-infer-triforce.exe tools\nq2-infer-triforce.c
if %errorlevel% neq 0 goto fail
gcc %CFLAGS% -pthread -o nq2-infer-triforce-dbg.exe tools\nq2-infer-triforce-dbg.c
if %errorlevel% neq 0 goto fail
echo built the fifteen experimental tools (see EXPERIMENTS-2026-11-24.md and PIPELINE-2026-11-24.md)
exit /b 0
:fail
echo build failed: is MinGW gcc on your PATH? (the engines need a POSIX-thread MinGW)
exit /b %errorlevel%
