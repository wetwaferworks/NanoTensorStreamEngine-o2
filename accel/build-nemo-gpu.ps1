# build-nemo-gpu.ps1 - builds the Little Nemo CUDA lane (accel/) and
# nemo-engine-cuda.exe. New binary names only, written under accel/bin
# and accel/obj; nothing in littlenemo/bin or littlenemo/obj is rebuilt.
# Run from the repository root. Requires gcc (MSYS2 UCRT64) and Python.
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $Root 'littlenemo\src\engine_gui.c'))) {
    # script lives in accel/, so parent is the repo root
    $Root = $PSScriptRoot + '\..'
}
Set-Location $Root

$F = '-O2','-std=c99','-Wall','-Wextra','-static','-D_FILE_OFFSET_BITS=64','-D_POSIX_C_SOURCE=200809L'
New-Item -ItemType Directory -Force accel\obj, accel\bin | Out-Null

$python = $null
foreach ($c in @('python','python3','py')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; if ($c -eq 'py') { $pythonArgs = @('-3') } else { $pythonArgs = @() }; break }
}
if (-not $python) { throw 'python not found on PATH' }

& $python @pythonArgs accel\mk_engine_gui_cuda.py littlenemo\src\engine_gui.c accel\engine_gui_cuda.c
if ($LASTEXITCODE) { throw 'generate' }

# Little Nemo objects, compiled into accel/obj so the official engine build is untouched.
gcc @F '-mavx2' '-c' '-o' 'accel\obj\ln_nn_simd.o' 'littlenemo\src\nn_simd.c'
if ($LASTEXITCODE) { throw 'nn_simd.c' }
foreach ($name in @('attn','forward','json','loader','mamba','nn','pool','tokenizer')) {
    gcc @F '-c' '-o' "accel\obj\ln_$name.o" "littlenemo\src\$name.c"
    if ($LASTEXITCODE) { throw "$name.c" }
}
$ln = @('attn','forward','json','loader','mamba','nn','nn_simd','pool','tokenizer') | ForEach-Object { "accel\obj\ln_$_.o" }

gcc @F '-Ilittlenemo\src' '-c' '-o' 'accel\obj\nemo_gpu.o' 'accel\nemo_gpu.c'
if ($LASTEXITCODE) { throw 'nemo_gpu.c' }
gcc @F '-Ilittlenemo\src' '-Iaccel' '-c' '-o' 'accel\obj\nemo_calib.o' 'accel\nemo_calib.c'
if ($LASTEXITCODE) { throw 'nemo_calib.c' }
gcc @F '-Ilittlenemo\src' '-c' '-o' 'accel\obj\nemo_predict.o' 'accel\nemo_predict.c'
if ($LASTEXITCODE) { throw 'nemo_predict.c' }
gcc @F '-Ilittlenemo\src' '-c' '-o' 'accel\obj\nemo_tokpipe.o' 'accel\nemo_tokpipe.c'
if ($LASTEXITCODE) { throw 'nemo_tokpipe.c' }
gcc @F '-Ilittlenemo\src' '-Iaccel' '-c' '-o' 'accel\obj\engine_gui_cuda.o' 'accel\engine_gui_cuda.c'
if ($LASTEXITCODE) { throw 'engine_gui_cuda.c' }

gcc -O2 -static -o accel\bin\nemo-engine-cuda.exe accel\obj\engine_gui_cuda.o accel\obj\nemo_gpu.o accel\obj\nemo_calib.o accel\obj\nemo_predict.o accel\obj\nemo_tokpipe.o @ln -lm -lpsapi
if ($LASTEXITCODE) { throw 'link engine' }
# windowless twin for the GUI shortcut (no console window pops up; same code)
gcc -O2 -static -mwindows -o accel\bin\nemo-engine-cuda-w.exe accel\obj\engine_gui_cuda.o accel\obj\nemo_gpu.o accel\obj\nemo_calib.o accel\obj\nemo_predict.o accel\obj\nemo_tokpipe.o @ln -lm -lpsapi
if ($LASTEXITCODE) { throw 'link engine-w' }

$testln = $ln | Where-Object { $_ -notmatch 'tokenizer' }
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\test_nemo_gpu_layers.c accel\obj\nemo_gpu.o @testln -o accel\bin\test_nemo_gpu_layers.exe -lm -lpsapi
if ($LASTEXITCODE) { throw 'test' }
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_pack_ident.c accel\obj\nemo_gpu.o @testln -o accel\bin\test_nemo_pack_ident.exe -lm -lpsapi
if ($LASTEXITCODE) { throw 'pack-ident test' }
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_prefill.c accel\obj\nemo_gpu.o @testln -o accel\bin\test_nemo_prefill.exe -lm -lpsapi
if ($LASTEXITCODE) { throw 'prefill test' }
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_hostsplit.c accel\obj\nemo_gpu.o accel\obj\nemo_predict.o @testln -o accel\bin\test_nemo_hostsplit.exe -lm -lpsapi
if ($LASTEXITCODE) { throw 'hostsplit test' }
# host-only: self-test slot bounds and the calibration kernel digest (compiles nemo_gpu.c itself)
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_selftest_slots.c @testln -o accel\bin\test_nemo_selftest_slots.exe -lm -lpsapi
if ($LASTEXITCODE) { throw 'selftest-slots test' }
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_tokpipe.c accel\obj\nemo_tokpipe.o accel\obj\ln_tokenizer.o accel\obj\ln_json.o accel\obj\ln_pool.o -o accel\bin\test_nemo_tokpipe.exe -lm
if ($LASTEXITCODE) { throw 'tokpipe test' }

'built accel\bin\nemo-engine-cuda.exe, nemo-engine-cuda-w.exe, test_nemo_gpu_layers.exe and test_nemo_pack_ident.exe'
