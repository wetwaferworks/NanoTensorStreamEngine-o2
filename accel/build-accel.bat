@echo off
rem NTSE-o2 accel lanes (research, scratch) - Windows, MinGW gcc.
rem Run from the REPO ROOT:  accel\build-accel.bat
rem Builds ONLY under accel\ (accel\bin, accel\obj). It never touches
rem the pure C build (build.bat, tools\, littlenemo\bin, littlenemo\obj).
rem No CUDA, OpenCL or Vulkan SDK is needed: the vendor runtimes are
rem loaded at run time, and the binaries run (on the CPU reference)
rem on machines with no GPU at all.
setlocal
set FLAGS=-O2 -std=c99 -Wall -Wextra -static -D_FILE_OFFSET_BITS=64
set SRC=accel\src\accel_core.c accel\src\accel_cuda.c accel\src\accel_opencl.c
if not exist accel\bin mkdir accel\bin
if not exist accel\obj mkdir accel\obj
for %%T in (guard kernels stream real linear plan) do (
    gcc %FLAGS% -o accel\bin\test_accel_%%T.exe accel\tests\test_accel_%%T.c %SRC% -lm
    if errorlevel 1 (echo build failed: test_accel_%%T & exit /b 1)
)
gcc %FLAGS% -o accel\bin\ntse-accel-bench.exe accel\tools\ntse-accel-bench.c %SRC% -lm
if errorlevel 1 (echo build failed: ntse-accel-bench & exit /b 1)
rem Little Nemo with its projections on an accel lane. Elfie's sources
rem are compiled unmodified into accel\obj; nn.c is compiled with its
rem two public dispatch names renamed so accel\littlenemo\nn_accel.c
rem can route full projections and keep nn.c's scalar kernel as the
rem golden fallback.
set NFLAGS=%FLAGS% -D_POSIX_C_SOURCE=200809L
gcc %NFLAGS% -mavx2 -c -o accel\obj\nn_simd.o littlenemo\src\nn_simd.c
if errorlevel 1 (echo build failed: nn_simd & exit /b 1)
for %%F in (forward attn mamba loader json pool tokenizer) do (
    gcc %NFLAGS% -c -o accel\obj\%%F.o littlenemo\src\%%F.c
    if errorlevel 1 (echo build failed: %%F & exit /b 1)
)
gcc %NFLAGS% -Dnemo_linear_bf16=nemo_cpu_linear_bf16 -Dnemo_linear_bf16_rows_auto=nemo_cpu_linear_bf16_rows_auto -c -o accel\obj\nn_cpu.o littlenemo\src\nn.c
if errorlevel 1 (echo build failed: nn_cpu & exit /b 1)
set NOBJ=accel\obj\forward.o accel\obj\attn.o accel\obj\mamba.o accel\obj\nn_cpu.o accel\obj\nn_simd.o accel\obj\loader.o accel\obj\json.o accel\obj\pool.o accel\obj\tokenizer.o
gcc %NFLAGS% -o accel\bin\test_nemo_forward_accel.exe accel\littlenemo\test_nemo_forward_accel.c accel\littlenemo\nn_accel.c %NOBJ% %SRC% -lm
if errorlevel 1 (echo build failed: test_nemo_forward_accel & exit /b 1)
echo built accel\bin: 6 accel suites, ntse-accel-bench, test_nemo_forward_accel
