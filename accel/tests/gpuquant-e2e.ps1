# gpuquant-e2e.ps1 - nq2-quantize --gpu against the CPU path, byte for byte.
# Co-implemented by Cortana at JD's direction.
#
# Run from the repo root after accel\build-gpuquant.bat:
#   powershell -NoProfile -File accel\tests\gpuquant-e2e.ps1 [-Seeds 4] [-Threads 1,4,12]
#   (-Threads is one comma-separated string, so it survives -File)
#   powershell -NoProfile -File accel\tests\gpuquant-e2e.ps1 -Shards a.safetensors,b.safetensors -Recipes supervl -Name sigma
#
# For every recipe and thread count it runs three quantizers on the
# same input:
#   ref   the plain CPU build at 1 thread (the reference bytes)
#   cpu   the plain CPU build at N threads
#   gpu   the GPU build with --gpu at N threads
# and, when -Verify is given, also --gpu-verify (every GPU search
# re-checked on the CPU inside the run). Pass means:
#   - the .nq2, .nq2book and .nq2manifest have the same SHA-256 as
#     ref, with the same exit code, and
#   - gpu (and verify) produce the same four files as cpu at the same
#     thread count, report included.
# The report alone may differ from ref between the serial and the
# parallel per-tensor writer (an existing CPU-side difference, see
# the note printed in the summary); that is listed, not failed. Prints one line per run and a summary;
# exit code 0 only if everything matched. -Prune deletes each run's
# .nq2 and .nq2book once hashed (for real models, to save disk).
param(
  [int]$Seeds = 4,
  [string]$Threads = '1,4,12',
  [string[]]$Shards = @(),
  [string[]]$Recipes = @(),
  [string]$Name = 'fixtures',
  [string]$Out = 'build\gpuquant\e2e',
  [string]$Shipped = '',
  [switch]$Verify,
  [switch]$Prune
)
$ErrorActionPreference = 'Stop'
$ThreadList = @($Threads -split '[, ]+' | ? { $_ } | % { [int]$_ })
$B = (Resolve-Path 'build\gpuquant').Path
$GPU = "$B\nq2-quantize.exe"; $CPU = "$B\nq2-quantize-cpu.exe"; $MK = "$B\gq_mkfixture.exe"
foreach ($x in $GPU, $CPU) { if (-not (Test-Path $x)) { throw "missing $x; run accel\build-gpuquant.bat" } }
$Root = "$Out\$Name"
New-Item -ItemType Directory -Force $Root | Out-Null
$Root = (Resolve-Path $Root).Path

$All = [ordered]@{
  'global'        = @()
  'global-rms'    = @('--norm', 'rms')
  'pt-flat-v2'    = @('--books', 'per-tensor', '--format', '2')
  'pt-flat-rms'   = @('--books', 'per-tensor', '--format', '3', '--norm', 'rms')
  'supervl-exact' = @('--books', 'per-tensor', '--norm', 'rms', '--pq', '4', '--raw-names', 'embeddings.weight,lm_head,gate.weight,pos_embed,mlp1.', '--raw-small', '1048576', '--skip-unsupported')
  'supervl'       = @('--books', 'per-tensor', '--norm', 'rms', '--pq', '4', '--raw-names', 'embed_tokens,lm_head,pos_embed', '--raw-small', '4096', '--skip-unsupported')
  'pq8'           = @('--books', 'per-tensor', '--format', '3', '--pq', '8')
  'pq4-small'     = @('--books', 'per-tensor', '--format', '3', '--pq', '4', '--tensor-sample-blocks', '500', '--entries', '16', '--iters', '3')
  'budgets'       = @('--books', 'per-tensor', '--format', '3', '--budgets')
  'residual'      = @('--books', 'per-tensor', '--format', '3', '--residual')
  'pt-b64'        = @('--books', 'per-tensor', '--format', '3', '--block', '64', '--entries', '64')
  'global-b1024'  = @('--block', '1024', '--entries', '32', '--sample-blocks', '3000')
}
if ($Recipes.Count -eq 0) { $Recipes = @($All.Keys | ? { $_ -ne 'supervl-exact' }) }
if ($Recipes.Count -eq 1 -and $Recipes[0] -match ',') { $Recipes = @($Recipes[0] -split ',') }

if ($Shards.Count -eq 0) {
  $sets = @()
  for ($s = 1; $s -le $Seeds; $s++) {
    $n = "$Root\normal-$s.safetensors"; $e = "$Root\edge-$s.safetensors"
    & $MK $n $s normal | Out-Null; & $MK $e (100 + $s) edge | Out-Null
    $sets += , @("seed$s", @($n, $e))
  }
  $bad = "$Root\bad.safetensors"; & $MK $bad 7 bad | Out-Null
  $sets += , @('bad', @($bad))
} else {
  $sets = @(, @($Name, @($Shards | % { (Resolve-Path $_).Path })))
}

function HashOf($dir, $exts) {
  $h = @()
  foreach ($ext in $exts) {
    $f = "$dir\pack.$ext"
    if (Test-Path $f) { $h += "$ext=" + (Get-FileHash $f -Algorithm SHA256).Hash } else { $h += "$ext=none" }
  }
  return ($h -join ' ')
}
$PackExt = @('nq2', 'nq2book', 'nq2manifest')
$AllExt = @('nq2', 'nq2book', 'nq2manifest', 'report.txt')
function Run($exe, $extra, $dir, $set, $recipe, $t) {
  $ErrorActionPreference = 'Continue'   # a refusal on stderr is a result here, not a script error
  if (Test-Path $dir) { Remove-Item -Recurse -Force $dir }
  New-Item -ItemType Directory -Force $dir | Out-Null
  $argv = @('--out', 'pack') + $All[$recipe] + @('--threads', "$t") + $extra + $set
  $sw = [Diagnostics.Stopwatch]::StartNew()
  Push-Location $dir
  try { & $exe @argv > stdout.txt 2> stderr.txt; $rc = $LASTEXITCODE } finally { Pop-Location }
  $sw.Stop()
  $pk = "rc=$rc " + (HashOf $dir $PackExt); $al = "rc=$rc " + (HashOf $dir $AllExt)
  if ($Prune) { Remove-Item "$dir\pack.nq2", "$dir\pack.nq2book" -ErrorAction SilentlyContinue }
  return @{ rc = $rc; pack = $pk; all = $al; secs = $sw.Elapsed.TotalSeconds; dir = $dir }
}

$pass = 0; $fail = 0; $repnote = 0; $lines = @()
function Mark($ok) { if ($ok) { $script:pass++; return 'MATCH' } else { $script:fail++; return 'DIFFER' } }
foreach ($set in $sets) {
  $sname = $set[0]; $files = $set[1]
  foreach ($r in $Recipes) {
    $base = "$Root\runs\$sname\$r"
    $ref = Run $CPU @() "$base\ref-t1" $files $r 1
    $line = "{0,-7} {1,-13} ref    t1  {2,7:N1}s rc={3}" -f $sname, $r, $ref.secs, $ref.rc
    $lines += $line; Write-Host $line
    if ($Shipped) {
      $sh = Run $Shipped @() "$base\shipped-t1" $files $r 1
      $line = "{0,-7} {1,-13} shipped t1 {2,7:N1}s {3}" -f $sname, $r, $sh.secs, (Mark ($sh.all -eq $ref.all))
      $lines += $line; Write-Host $line
    }
    foreach ($t in $ThreadList) {
      if ($t -eq 1) { $cres = $ref } else { $cres = Run $CPU @() "$base\cpu-t$t" $files $r $t }
      $note = ''
      if ($cres.all -ne $ref.all -and $cres.pack -eq $ref.pack) { $note = '(report text differs from t1)'; $repnote++ }
      $line = "{0,-7} {1,-13} cpu    t{2,-3} {3,7:N1}s rc={4} {5} {6}" -f $sname, $r, $t, $cres.secs, $cres.rc, (Mark ($cres.pack -eq $ref.pack)), $note
      $lines += $line; Write-Host $line
      $modes = @(, @('gpu', @('--gpu')))
      if ($Verify) { $modes += , @('verify', @('--gpu-verify')) }
      foreach ($m in $modes) {
        $res = Run $GPU $m[1] "$base\$($m[0])-t$t" $files $r $t
        $ok = ($res.pack -eq $ref.pack) -and ($res.all -eq $cres.all)
        $g = Get-Content "$($res.dir)\stdout.txt" | ? { $_ -match '^gpu: \d+ nearest' }
        $note = $(if ($g) { ($g -replace '^gpu: ', '') } else { '' })
        $line = "{0,-7} {1,-13} {2,-6} t{3,-3} {4,7:N1}s rc={5} {6} {7}" -f $sname, $r, $m[0], $t, $res.secs, $res.rc, (Mark $ok), $note
        $lines += $line; Write-Host $line
      }
    }
  }
}
if ($repnote) { $lines += "note: $repnote CPU runs wrote a report that differs from the 1-thread report while the pack files match (the parallel per-tensor writer leaves out the product-quantization line); GPU runs are compared with the CPU run at the same thread count, report included" }
$sum = "RESULT: $pass match, $fail differ"
if ($fail -eq 0) { $sum = "RESULT: ALL MATCH ($pass runs byte-identical to the 1-thread CPU reference)" }
$lines += $sum; Write-Host $sum
$lines | Set-Content "$Root\e2e-summary.txt"
if ($fail -ne 0) { exit 1 } else { exit 0 }
