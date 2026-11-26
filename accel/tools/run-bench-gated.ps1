# run-bench-gated.ps1 - NTSE-o2 accel benchmark receipts, gated on the
# Super VL HQ conversion. Research/scratch; co-implemented by Cortana.
# Run from the REPO ROOT (detached):
#   powershell -NoProfile -ExecutionPolicy Bypass -File accel\tools\run-bench-gated.ps1
# Waits until the converter process is gone, then runs each lane's
# sections as separate short BelowNormal processes (one log each),
# sampling peak working set and peak device memory (nvidia-smi), with
# pauses between runs so CPU baselines running alongside keep room.
param(
  [int]$GatePid = 7644,
  [string]$R = 'results-2026-11-26',   # relative to the repo root
  [int]$Reps = 20,
  [int]$PauseSec = 20,
  [string]$Only = '',          # comma list of run names, '' = all
  [string]$Tag = 'bench'       # output folder under accel\logs
)
$ErrorActionPreference = 'Continue'
$root = Get-Location
$out = Join-Path $root "accel\logs\$Tag"
New-Item -ItemType Directory -Force $out | Out-Null
$rcpt = Join-Path $out 'RECEIPTS.txt'
function Log($m) { $l = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ET  $m"; Add-Content -Path $rcpt -Value $l -Encoding ascii }
$conv = "$R\super-vl-nq2-hq\convert.log"
if ($GatePid -gt 0) { Log "gate: waiting for pid $GatePid (Super VL HQ conversion) to exit" } else { Log "gate: none requested" }
$deadline = (Get-Date).AddHours(8)
while ($GatePid -gt 0 -and (Get-Process -Id $GatePid -ErrorAction SilentlyContinue)) {
  if ((Get-Date) -gt $deadline) { Log "gate: still running after 8 h; giving up, no benchmarks run"; exit 2 }
  Start-Sleep 60
}
if ($GatePid -gt 0) {
  Log "gate: pid $GatePid gone; convert.log tail:"
  Get-Content $conv -Tail 4 -ErrorAction SilentlyContinue | ForEach-Object { Log "   | $_" }
  Start-Sleep 30
}
# ---- receipts header ----
Log "rev: $(git rev-parse HEAD) (branch $(git rev-parse --abbrev-ref HEAD)); tree: $((git status --short) -join ' ; ') (accel/ untracked, nothing tracked modified)"
$pack = "$R\pack\sigma-full-b256-v2run.nq2"; $book = "$R\pack\sigma-full-b256-v2run.nq2book"
foreach ($f in $pack, $book) { $h = Get-FileHash -Algorithm SHA256 $f; Log "pack-id: $(Split-Path $f -Leaf) sha256 $($h.Hash.ToLower()) bytes $((Get-Item $f).Length)" }
Log "compiler: $((& gcc --version | Select-Object -First 1)) flags -O2 -std=c99 -static"
Log "gpu: $((& nvidia-smi --query-gpu=name,driver_version,memory.total,pstate --format=csv,noheader) -join ' | ')"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
$os = Get-CimInstance Win32_OperatingSystem
Log "cpu: $($cpu.Name.Trim()) $($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T; RAM $([int]($os.TotalVisibleMemorySize/1024)) MiB total, $([int]($os.FreePhysicalMemory/1024)) MiB free; other jobs: $((Get-Process nq2-*,pchie* -ErrorAction SilentlyContinue | ForEach-Object { "$($_.ProcessName)/$($_.Id)/$($_.PriorityClass)" }) -join ', ')"
$shards = "$R\model\sigma-3b\model-00001-of-00002.safetensors $R\model\sigma-3b\model-00002-of-00002.safetensors"
$runs = @(
  @('cuda-abcd',   "--lane cuda --cpu --reps $Reps --sections abcd"),
  @('cuda-s',      "--lane cuda --sections s"),
  @('cuda-p',      "--lane cuda --cpu --sections p --pack $pack $book $shards"),
  @('gpucl-abcd',  "--lane gpu-cl --reps $Reps --sections abcd"),
  @('gpucl-s',     "--lane gpu-cl --sections s"),
  @('gpucl-p',     "--lane gpu-cl --sections p --pack $pack $book $shards"),
  @('igpu-abcd',   "--lane igpu --reps $Reps --sections abcd"),
  @('igpu-s',      "--lane igpu --sections s"),
  @('igpu-p',      "--lane igpu --sections p --pack $pack $book $shards")
)
function Num($t) { $d = 0.0; if ([double]::TryParse(($t -replace '[^0-9.]', ''), [ref]$d)) { $d } else { -1 } }
function GpuSample { $v = & nvidia-smi --query-gpu=memory.used,temperature.gpu,clocks.sm,power.draw --format=csv,noheader,nounits 2>$null; if ($v) { $f = ($v | Select-Object -First 1).Split(','); @((Num $f[0]), (Num $f[1]), (Num $f[2]), (Num $f[3])) } else { @(-1, -1, -1, -1) } }
function VramUsed { $v = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null; if ($v) { [int]($v | Select-Object -First 1) } else { -1 } }
$want = @(); if ($Only) { $want = $Only.Split(',') }
foreach ($run in $runs) {
  $name = $run[0]; $a = $run[1]; $log = Join-Path $out "$name.log"
  if ($want.Count -gt 0 -and -not ($want -contains $name)) { continue }
  $g0 = GpuSample; $v0 = $g0[0]; $tmax = $g0[1]; $clkmin = 1e9; $pmax = 0
  $t0 = Get-Date
  $p = Start-Process cmd.exe -ArgumentList '/c', "accel\bin\ntse-accel-bench.exe $a > `"$log`" 2>&1" -WorkingDirectory $root -PassThru -WindowStyle Hidden
  try { $p.PriorityClass = 'BelowNormal' } catch {}
  $peakWS = 0; $peakV = $v0; $minFree = 1e9
  while (-not $p.HasExited) {
    $c = Get-Process ntse-accel-bench -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($c) { try { if ($c.PriorityClass -ne 'BelowNormal') { $c.PriorityClass = 'BelowNormal' } } catch {}; $c.Refresh(); $w = [int]($c.PeakWorkingSet64/1MB); if ($w -gt $peakWS) { $peakWS = $w } }
    $g = GpuSample; if ($g[0] -gt $peakV) { $peakV = $g[0] }; if ($g[1] -gt $tmax) { $tmax = $g[1] }; if ($g[2] -gt 0 -and $g[2] -lt $clkmin) { $clkmin = $g[2] }; if ($g[3] -gt $pmax) { $pmax = $g[3] }
    $fr = [int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024); if ($fr -lt $minFree) { $minFree = $fr }
    if ($fr -lt 400 -and $c) { Stop-Process -Id $c.Id -Force; Log "$name KILLED: free RAM $fr MiB < 400"; break }
    if (((Get-Date) - $t0).TotalMinutes -gt 30) { if ($c) { Stop-Process -Id $c.Id -Force }; Log "$name KILLED: over 30 min"; break }
    Start-Sleep -Milliseconds 1000
  }
  $p.WaitForExit()
  Log ("run {0}: args [{1}] exit {2} wall {3:N1}s peak_ws_MiB {4} vram_used_MiB before {5} peak {6} (delta {7}) min_free_ram_MiB {8} gpu_temp_C start {9} max {10} sm_clock_min_MHz {11} power_max_W {12}" -f $name, $a, $p.ExitCode, ((Get-Date)-$t0).TotalSeconds, $peakWS, $v0, $peakV, ($peakV-$v0), $minFree, $g0[1], $tmax, $clkmin, $pmax)
  Start-Sleep $PauseSec
}
Log "done: receipts in accel\logs\$Tag\*.log (TSV lines start with R)"
