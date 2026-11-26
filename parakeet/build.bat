@echo off
rem parakeet-c build (MSYS2 UCRT64 gcc). Pure C11 + WASAPI. No Python, no WSL.
setlocal
if "%GCC%"=="" set GCC=gcc
if exist C:\msys64\ucrt64\bin set PATH=C:\msys64\ucrt64\bin;%PATH%
cd /d %~dp0
if not exist bin mkdir bin
set CF=-O3 -march=native -mavx2 -mfma -std=c11 -Wall -Wno-unused-parameter -Wno-stringop-truncation
set SRC=src\util.c src\gemm.c src\mel.c src\audio.c src\spm.c src\model.c src\mic_wasapi.c
%GCC% %CF% %SRC% src\main.c -o bin\parakeet-c.exe -lole32 -luuid -static || exit /b 1
%GCC% %CF% tests\test_units.c %SRC% -o bin\parakeet-tests.exe -lole32 -luuid -static || exit /b 1
echo built bin\parakeet-c.exe and bin\parakeet-tests.exe
