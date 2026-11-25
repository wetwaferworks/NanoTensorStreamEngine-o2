@echo off
rem Nano Quant Engine, one-click build, Windows with MinGW gcc
rem (MSYS2 works well). Builds every tool plus Little Nemo. Run
rem from the repo root:
rem
rem     build-all.bat
rem
rem build.bat already builds all 23 tools and Little Nemo, so
rem this script is just the one-click name for it. (It used to
rem run build-experiments.bat afterwards as well, which
rem recompiled the research tools a second time for no
rem benefit... Archie review P2-1, 2026-11-24.
rem build-experiments.bat remains as the research-subset build.)
rem
rem Everything is statically linked, so no MinGW runtime DLLs
rem are needed. The inference engines need a POSIX-thread MinGW
rem (winpthreads) for their thread pools.
call build.bat
if %errorlevel% neq 0 exit /b %errorlevel%
echo build-all: done (23 tools + Little Nemo)
