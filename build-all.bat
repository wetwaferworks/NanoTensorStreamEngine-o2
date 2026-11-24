@echo off
rem Nano Quant Engine, one-click build, Windows with MinGW gcc
rem (MSYS2 works well). Builds every tool: the six production
rem tools and the fifteen experimental ones. Run from the repo
rem root:
rem
rem     build-all.bat
rem
rem Everything is statically linked, so no MinGW runtime DLLs
rem are needed. The inference engines need a POSIX-thread MinGW
rem (winpthreads) for their thread pools.
call build.bat
if %errorlevel% neq 0 exit /b %errorlevel%
call build-experiments.bat
if %errorlevel% neq 0 exit /b %errorlevel%
echo build-all: done (21 tools)
