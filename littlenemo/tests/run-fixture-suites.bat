@echo off
rem Runs the Little Nemo suites that need crafted or partial checkpoints:
rem test_hostile, test_coherence and test_prefill.
rem
rem   run-fixture-suites.bat <checkpoint_dir> [fixture_dir]
rem
rem Run from the littlenemo directory after build.bat. The fixtures are
rem made by bin\mk_fixtures.exe under fixture_dir (default
rem %TEMP%\nemo-fixtures): tiny crafted files, one sparse file with no
rem model data, and hard links to the first two shards of the checkpoint.
rem Keep fixture_dir outside the repository. Hard links need fixture_dir on
rem the same volume as the checkpoint.
setlocal enabledelayedexpansion
if "%~1"=="" (echo usage: run-fixture-suites.bat ^<checkpoint_dir^> [fixture_dir] & exit /b 2)
set CKPT=%~1
set FX=%~2
if "%FX%"=="" set FX=%TEMP%\nemo-fixtures
set FAILED=0

bin\mk_fixtures.exe "%CKPT%" "%FX%"
if errorlevel 1 (echo fixture build failed & exit /b 1)

for /d %%D in ("%FX%\hostile\refuse-*") do (
    bin\test_hostile.exe refuse "%%D" >nul
    if errorlevel 1 (echo FAIL hostile %%~nxD & set FAILED=1) else (echo ok   hostile %%~nxD)
)
bin\test_hostile.exe i32 "%FX%\hostile\i32"
if errorlevel 1 set FAILED=1

bin\test_coherence.exe "%FX%\partial" "%FX%\dtype"
if errorlevel 1 set FAILED=1

bin\test_prefill.exe "%CKPT%"
if errorlevel 1 set FAILED=1

if "%FAILED%"=="1" (echo fixture suites: FAILURES & exit /b 1)
echo fixture suites: ALL PASS
