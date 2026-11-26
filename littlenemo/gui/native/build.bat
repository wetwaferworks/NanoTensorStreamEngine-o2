@echo off
REM Build with the existing UCRT64 gcc. No installs.
setlocal
cd /d "%~dp0"
if not defined CC set CC=gcc
set CFLAGS=-O2 -std=c11 -Wall -Wextra -Werror -fPIC -DNEMO_SCALE_BUILD
%CC% %CFLAGS% -o test_nemo_scale.exe test_nemo_scale.c nemo_scale.c -lm
if errorlevel 1 exit /b 1
%CC% %CFLAGS% -shared -o nemo_scale.dll nemo_scale.c -lm
if errorlevel 1 exit /b 1
echo built in %CD%
dir /b test_nemo_scale.exe nemo_scale.dll
