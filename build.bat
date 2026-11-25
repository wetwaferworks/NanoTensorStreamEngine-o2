@echo off
rem NanoQuatTwo build, Windows with MinGW gcc. Run from the repo root.
rem Builds everything: all 23 tools in tools\ and the Little Nemo
rem engine with its test suites (littlenemo\bin\), statically linked
rem so no MinGW runtime DLLs are needed.
rem Note: the threaded tools (nq2-infer, nq2-infer-lab,
rem nq2-infer-triforce, nq2-infer-triforce-dbg, nq2-streamtest) need
rem a POSIX-thread MinGW (winpthreads); the rest build with either
rem thread model. Little Nemo uses native Win32 threads and builds
rem with any MinGW.
setlocal
set FLAGS=-O2 -std=c99 -Wall -Wextra -static -D_FILE_OFFSET_BITS=64
for %%T in (quantize run audit frameprobe errprobe tensorstat entries hadamard pq additive framed fused correct rpq dc dc-adv logitcmp) do (
    gcc %FLAGS% -o nq2-%%T.exe tools\nq2-%%T.c -lm
    if errorlevel 1 (echo build failed: nq2-%%T - is MinGW gcc on your PATH? & exit /b 1)
)
rem rsqrtbench is a benchmark: its AVX2/AVX-512 sections are
rem guarded by __AVX2__/__AVX512F__ and silently vanish without
rem the flags (Archie P2-3). Build it like build-experiments.bat.
gcc %FLAGS% -O3 -march=native -o nq2-rsqrtbench.exe tools\nq2-rsqrtbench.c -lm
if errorlevel 1 (echo build failed: nq2-rsqrtbench & exit /b 1)
for %%T in (infer infer-lab infer-triforce infer-triforce-dbg streamtest) do (
    gcc %FLAGS% -pthread -o nq2-%%T.exe tools\nq2-%%T.c -lm
    if errorlevel 1 (echo build failed: nq2-%%T - the threaded tools need a POSIX-thread MinGW & exit /b 1)
)
cd littlenemo
call build.bat
if errorlevel 1 (echo build failed: littlenemo & exit /b 1)
cd ..
echo built all 23 nq2 tools and Little Nemo (littlenemo\bin\)
