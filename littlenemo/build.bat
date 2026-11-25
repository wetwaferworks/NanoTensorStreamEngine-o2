@echo off
rem Little Nemo build, Windows with MinGW gcc. Run from the littlenemo
rem directory. Builds everything: the engine, the checkpoint
rem inspector, and every test suite, into bin\.
setlocal
if not exist bin mkdir bin
if not exist obj mkdir obj
set FLAGS=-O2 -std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L -D_FILE_OFFSET_BITS=64 -static
rem nn.c alone gets AVX2 codegen. Its kernels are reached only
rem through a runtime CPUID check and only when the engine is told
rem --simd, so the exe still runs on older hardware, which stays
rem on the scalar path.
gcc %FLAGS% -mavx2 -c -o obj\nn.o src\nn.c
if errorlevel 1 (echo build failed: nn & exit /b 1)
for %%F in (forward attn mamba loader json pool tokenizer) do (
    gcc %FLAGS% -c -o obj\%%F.o src\%%F.c
    if errorlevel 1 (echo build failed: %%F & exit /b 1)
)
set LIB=obj\forward.o obj\attn.o obj\mamba.o obj\nn.o obj\loader.o obj\json.o obj\pool.o obj\tokenizer.o
gcc %FLAGS% -o bin\nemo-engine.exe src\engine.c %LIB% -lm -lpsapi
if errorlevel 1 (echo build failed: nemo-engine & exit /b 1)
gcc %FLAGS% -o bin\nemo-inspect.exe src\nemo.c %LIB% -lm
if errorlevel 1 (echo build failed: nemo-inspect & exit /b 1)
for %%T in (test_mamba test_attn test_loader test_forward test_tokenizer test_guards test_hostile test_coherence test_par test_simd) do (
    gcc %FLAGS% -o bin\%%T.exe tests\%%T.c %LIB% -lm
    if errorlevel 1 (echo build failed: %%T & exit /b 1)
)
echo built bin\nemo-engine.exe bin\nemo-inspect.exe and 10 test suites in bin\
