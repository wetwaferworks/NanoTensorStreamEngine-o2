# build-nemo-expand.ps1 - builds the expanded-pack (refinement stage) tools
# and a CUDA engine into a separate output directory, so accel\bin,
# accel\obj and every earlier build stay untouched.
#   powershell -File accel\build-nemo-expand.ps1 [-Out build\expand]
# Run from anywhere; paths are resolved from the repository root.
# Requires gcc (MSYS2 UCRT64) and Python.
param([string]$Out = 'build\expand')
$ErrorActionPreference = 'Stop'
Set-Location (Split-Path -Parent $PSScriptRoot)

$F = '-O2','-std=c99','-Wall','-Wextra','-static','-D_FILE_OFFSET_BITS=64','-D_POSIX_C_SOURCE=200809L'
$obj = Join-Path $Out 'obj'
$bin = Join-Path $Out 'bin'
New-Item -ItemType Directory -Force $obj, $bin | Out-Null

$python = $null
foreach ($c in @('python','python3','py')) {
    $cmd = Get-Command $c -ErrorAction SilentlyContinue
    if ($cmd) { $python = $cmd.Source; if ($c -eq 'py') { $pythonArgs = @('-3') } else { $pythonArgs = @() }; break }
}
if (-not $python) { throw 'python not found on PATH' }

function cc([string]$what) { if ($LASTEXITCODE) { throw $what } }

# generated engine source goes into the output directory, not accel\
$gen = Join-Path $obj 'engine_gui_cuda.c'
& $python @pythonArgs accel\mk_engine_gui_cuda.py littlenemo\src\engine_gui.c $gen; cc 'generate'

gcc @F '-mavx2' '-c' '-o' "$obj\ln_nn_simd.o" 'littlenemo\src\nn_simd.c'; cc 'nn_simd.c'
foreach ($name in @('attn','forward','json','loader','mamba','nn','pool','tokenizer')) {
    gcc @F '-c' '-o' "$obj\ln_$name.o" "littlenemo\src\$name.c"; cc "$name.c"
}
$ln = @('attn','forward','json','loader','mamba','nn','nn_simd','pool','tokenizer') | ForEach-Object { "$obj\ln_$_.o" }
$testln = $ln | Where-Object { $_ -notmatch 'tokenizer' }

gcc @F '-mavx2' '-mfma' '-Ilittlenemo\src' '-c' '-o' "$obj\nemo_gpu.o" 'accel\nemo_gpu.c'; cc 'nemo_gpu.c'
gcc @F '-Ilittlenemo\src' '-Iaccel' '-c' '-o' "$obj\nemo_calib.o" 'accel\nemo_calib.c'; cc 'nemo_calib.c'
gcc @F '-Ilittlenemo\src' '-c' '-o' "$obj\nemo_predict.o" 'accel\nemo_predict.c'; cc 'nemo_predict.c'
gcc @F '-Ilittlenemo\src' '-c' '-o' "$obj\nemo_tokpipe.o" 'accel\nemo_tokpipe.c'; cc 'nemo_tokpipe.c'
gcc @F '-Ilittlenemo\src' '-Iaccel' '-c' '-o' "$obj\engine_gui_cuda.o" $gen; cc 'engine_gui_cuda.c'
$eng = @("$obj\engine_gui_cuda.o","$obj\nemo_gpu.o","$obj\nemo_calib.o","$obj\nemo_predict.o","$obj\nemo_tokpipe.o")
gcc -O2 -static -o "$bin\nemo-engine-cuda-x.exe" @eng @ln -lm -lpsapi; cc 'link engine'

gcc @F '-Ilittlenemo\src' '-Iaccel' accel\test_nemo_gpu_layers.c "$obj\nemo_gpu.o" @testln -o "$bin\test_nemo_gpu_layers.exe" -lm -lpsapi; cc 'layers test'
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_pack_ident.c "$obj\nemo_gpu.o" @testln -o "$bin\test_nemo_pack_ident.exe" -lm -lpsapi; cc 'pack-ident test'
gcc @F '-Ilittlenemo\src' '-Iaccel' accel\tests\test_nemo_selftest_slots.c @testln -o "$bin\test_nemo_selftest_slots.exe" -lm -lpsapi; cc 'selftest-slots test'

# expansion tools: CPU sensitivity probe, residual writer, pack merger
gcc @F '-mavx2' '-mfma' '-Ilittlenemo\src' littlenemo\expand\ln_sens.c @ln -o "$bin\ln_sens.exe" -lm; cc 'ln_sens.c'
gcc @F '-Ilittlenemo\src' littlenemo\expand\ln_resid.c @ln -o "$bin\ln_resid.exe" -lm; cc 'ln_resid.c'
gcc @F tools\nq2-merge-refine.c -o "$bin\nq2-merge-refine.exe"; cc 'nq2-merge-refine.c'
gcc @F '-mavx2' '-mfma' '-Ilittlenemo\src' littlenemo\expand\ln_headbench.c @ln -o "$bin\ln_headbench.exe" -lm; cc 'ln_headbench.c'

"built $bin\nemo-engine-cuda-x.exe, the layer, pack-ident and slot tests, ln_sens.exe, ln_resid.exe, nq2-merge-refine.exe and ln_headbench.exe"
