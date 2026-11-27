@echo off
rem nq2-quantize GPU lane - Windows, MinGW gcc. Run from the REPO ROOT:
rem     accel\build-gpuquant.bat
rem Builds ONLY into build\gpuquant (git-ignored). It never touches
rem the pure C build (build.bat and the tools at the repo root).
rem No CUDA SDK is needed to build: the CUDA driver and NVRTC are
rem loaded at run time (see accel\src\gq_cuda.c), and without them
rem `--gpu` reports why and exits.
rem   nq2-quantize.exe      the quantizer with the --gpu lane
rem   nq2-quantize-cpu.exe  the plain build, the same as build.bat's
rem   test_gq_assign.exe    GPU against CPU search and k-means
rem   gq_mkfixture.exe      synthetic safetensors for the e2e test
setlocal
set FLAGS=-O2 -std=c99 -Wall -Wextra -static -D_FILE_OFFSET_BITS=64
set OUT=build\gpuquant
if not exist build mkdir build
if not exist %OUT% mkdir %OUT%
gcc %FLAGS% -DNQ2_GPU -Iaccel\src -o %OUT%\nq2-quantize.exe tools\nq2-quantize.c accel\src\gq_cuda.c -lm
if errorlevel 1 (echo build failed: nq2-quantize with --gpu & exit /b 1)
gcc %FLAGS% -o %OUT%\nq2-quantize-cpu.exe tools\nq2-quantize.c -lm
if errorlevel 1 (echo build failed: nq2-quantize-cpu & exit /b 1)
gcc %FLAGS% -Wno-unused-function -DNQ2_GPU -Iaccel\src -o %OUT%\test_gq_assign.exe accel\tests\test_gq_assign.c accel\src\gq_cuda.c -lm
if errorlevel 1 (echo build failed: test_gq_assign & exit /b 1)
gcc %FLAGS% -o %OUT%\gq_mkfixture.exe accel\tests\gq_mkfixture.c
if errorlevel 1 (echo build failed: gq_mkfixture & exit /b 1)
echo built %OUT%: nq2-quantize (--gpu), nq2-quantize-cpu, test_gq_assign, gq_mkfixture
