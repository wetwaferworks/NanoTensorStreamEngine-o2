@echo off
rem Little Nemo build, Windows with MinGW gcc. Run from the littlenemo
rem directory. Builds everything: the engine, the checkpoint
rem inspector, and every test suite, into bin\.
setlocal
if not exist bin mkdir bin
set FLAGS=-O2 -std=c99 -Wall -Wextra -D_POSIX_C_SOURCE=200809L
set SRC=src\forward.c src\attn.c src\mamba.c src\nn.c src\loader.c src\json.c src\pool.c
gcc %FLAGS% -o bin\nemo-engine.exe src\engine.c src\tokenizer.c %SRC% -lm
if errorlevel 1 (echo build failed: nemo-engine & exit /b 1)
gcc %FLAGS% -o bin\nemo-inspect.exe src\nemo.c src\tokenizer.c %SRC% -lm
if errorlevel 1 (echo build failed: nemo-inspect & exit /b 1)
for %%T in (test_mamba test_attn test_loader test_forward test_tokenizer test_guards test_hostile test_coherence test_par) do (
    gcc %FLAGS% -o bin\%%T.exe tests\%%T.c src\tokenizer.c %SRC% -lm
    if errorlevel 1 (echo build failed: %%T & exit /b 1)
)
echo built bin\nemo-engine.exe bin\nemo-inspect.exe and 9 test suites in bin\
