@echo off
rem NanoQuatTwo build, Windows with MinGW gcc. Run from the repo root.
rem Builds all six tools, statically linked so no MinGW runtime
rem DLLs are needed. Note: nq2-infer needs a POSIX-thread MinGW
rem (winpthreads) for its thread pool; the other three build with
rem either thread model. Prebuilt exes are in bin\win64 either way.
gcc -O2 -std=c99 -Wall -Wextra -static -o nq2-quantize.exe tools\nq2-quantize.c
if %errorlevel% neq 0 (
  echo build failed: is MinGW gcc on your PATH?
  exit /b %errorlevel%
)
gcc -O2 -std=c99 -Wall -Wextra -static -o nq2-run.exe tools\nq2-run.c
if %errorlevel% neq 0 (
  echo build failed: is MinGW gcc on your PATH?
  exit /b %errorlevel%
)
gcc -O2 -std=c99 -Wall -Wextra -static -o nq2-audit.exe tools\nq2-audit.c
if %errorlevel% neq 0 (
  echo build failed: is MinGW gcc on your PATH?
  exit /b %errorlevel%
)
gcc -O2 -std=c99 -Wall -Wextra -pthread -static -o nq2-infer.exe tools\nq2-infer.c
if %errorlevel% neq 0 (
  echo build failed on nq2-infer: it needs a POSIX-thread MinGW
  exit /b %errorlevel%
)
gcc -O2 -std=c99 -Wall -Wextra -static -o nq2-frameprobe.exe tools\nq2-frameprobe.c
if %errorlevel% neq 0 (
  echo build failed: is MinGW gcc on your PATH?
  exit /b %errorlevel%
)
gcc -O2 -std=c99 -Wall -Wextra -static -o nq2-errprobe.exe tools\nq2-errprobe.c
if %errorlevel% neq 0 (
  echo build failed: is MinGW gcc on your PATH?
  exit /b %errorlevel%
)
echo built nq2-quantize.exe nq2-run.exe nq2-audit.exe nq2-infer.exe nq2-frameprobe.exe nq2-errprobe.exe
