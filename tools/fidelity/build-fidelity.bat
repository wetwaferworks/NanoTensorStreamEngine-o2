@echo off
rem Fidelity lab build (Windows, MinGW gcc with POSIX threads). Run from
rem the repository root. Outputs go to build\fidelity, which git ignores.
rem The lab tools build with AVX2/FMA codegen for speed; results do not
rem depend on the thread count. nq2-infer-lab-std.exe is the patched lab
rem engine built with the repository's standard experimental flags, for
rem checking the pinned anchors.
setlocal
if not exist build\fidelity mkdir build\fidelity
set FAST=-O3 -std=c99 -Wall -Wextra -static -mavx2 -mfma -ffp-contract=fast -pthread
set STD=-O2 -std=c99 -Wall -Wextra -static -pthread
gcc %FAST% -o build\fidelity\fid-quant.exe tools\fidelity\fid-quant.c -lm
if errorlevel 1 goto fail
gcc %FAST% -o build\fidelity\fid-sigma.exe tools\fidelity\fid-sigma.c -lm
if errorlevel 1 goto fail
gcc %FAST% -o build\fidelity\fid-hess.exe tools\fidelity\fid-hess.c -lm
if errorlevel 1 goto fail
gcc -O2 -std=c99 -Wall -Wextra -static -D_POSIX_C_SOURCE=200809L -o build\fidelity\fid-tok.exe tools\fidelity\fid-tok.c littlenemo\src\tokenizer.c littlenemo\src\json.c
if errorlevel 1 goto fail
gcc %FAST% -o build\fidelity\nq2-infer-lab-fid.exe tools\nq2-infer-lab.c -lm
if errorlevel 1 goto fail
gcc %STD% -o build\fidelity\nq2-infer-lab-std.exe tools\nq2-infer-lab.c -lm
if errorlevel 1 goto fail
echo built the fidelity lab tools into build\fidelity
exit /b 0
:fail
echo fidelity build failed
exit /b 1
