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
set FLAGS=-O2 -std=c99 -Wall -Wextra -static
for %%T in (quantize run audit frameprobe errprobe tensorstat entries hadamard pq additive framed fused rsqrtbench correct rpq dc dc-adv logitcmp) do (
    gcc %FLAGS% -o nq2-%%T.exe tools\nq2-%%T.c -lm
    if errorlevel 1 (echo build failed: nq2-%%T - is MinGW gcc on your PATH? & exit /b 1)
)
for %%T in (infer infer-lab infer-triforce infer-triforce-dbg streamtest) do (
    gcc %FLAGS% -pthread -o nq2-%%T.exe tools\nq2-%%T.c -lm
    if errorlevel 1 (echo build failed: nq2-%%T - the threaded tools need a POSIX-thread MinGW & exit /b 1)
)
cd littlenemo
call build.bat
if errorlevel 1 (echo build failed: littlenemo & exit /b 1)
cd ..
echo built all 23 nq2 tools and Little Nemo (littlenemo\bin\)
