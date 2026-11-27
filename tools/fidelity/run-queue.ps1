# Fidelity lab job queue (run from the repository root).
#
# Runs the steps of a queue file in order, one at a time, and respects
# the machine's shared-use rules:
#   - while the SuperVL quantize (PID given by -Quantize) is alive, steps
#     run at BelowNormal priority with -ThreadsShared threads;
#   - after it exits and while testdata\postpack.ps1 is still running
#     (its log has no "POSTPACK DONE"), nothing runs: a step that is
#     running when that window opens is stopped and rerun later from the
#     start (every step is restartable);
#   - after that, steps run at BelowNormal with -ThreadsFull threads.
# Results do not depend on the thread count, so a rerun with another
# count gives the same bytes.
#
# Queue file lines: NAME<TAB>COMMAND. {T} in COMMAND becomes the thread
# count. "remove-dir PATH" deletes a variant directory under
# data\fidelity\variants (nothing else). A step is skipped when
# data\fidelity\runs\NAME.done exists. Lines starting with # are notes.
param(
  [Parameter(Mandatory=$true)][string]$Queue,
  [int]$Quantize = 30112,
  [int]$After = 0,
  [int]$ThreadsShared = 8,
  [int]$ThreadsFull = 12
)
$ErrorActionPreference = 'Stop'
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
$runs = 'data\fidelity\runs'
New-Item -ItemType Directory -Force $runs | Out-Null
$log = "$runs\queue.log"
$pplog = 'data\postpack\postpack.log'
function Log($t) { Add-Content $log ("[{0:yyyy-MM-dd HH:mm:ss}] {1}" -f (Get-Date), $t) }
function Gate {
  if (Get-Process -Id $Quantize -ErrorAction SilentlyContinue) { return 'shared' }
  $pp = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { $_.CommandLine -match 'postpack\.ps1' }
  if ($pp) {
    if ((Test-Path $pplog) -and (Select-String -Path $pplog -Pattern 'POSTPACK DONE' -Quiet)) { return 'full' }
    return 'closed'
  }
  return 'full'
}
function WaitOpen {
  $said = $false
  while ((Gate) -eq 'closed') {
    if (-not $said) { Log 'postpack is running: waiting'; $said = $true }
    Start-Sleep 30
  }
}
if ($After -gt 0) {
  Log "waiting for PID $After"
  while (Get-Process -Id $After -ErrorAction SilentlyContinue) { Start-Sleep 20 }
}
Log "queue $Queue started"
foreach ($line in Get-Content $Queue) {
  if ($line -match '^\s*(#|$)') { continue }
  $parts = $line -split "`t", 2
  $name = $parts[0].Trim(); $cmd = $parts[1].Trim()
  if (Test-Path "$runs\$name.done") { continue }
  if ($cmd -match '^remove-dir\s+(.+)$') {
    $d = $Matches[1].Trim()
    $full = [IO.Path]::GetFullPath($d)
    $root = [IO.Path]::GetFullPath('data\fidelity\variants')
    if (-not $full.StartsWith($root + '\')) { Log "$name refused: $d is outside data\fidelity\variants"; continue }
    if (Test-Path $full) { Remove-Item -Recurse -Force $full }
    New-Item -ItemType File "$runs\$name.done" | Out-Null
    Log "$name removed $d"
    continue
  }
  $exe, $rest = $cmd -split '\s+', 2
  while ($true) {
    WaitOpen
    $g = Gate
    $t = if ($g -eq 'shared') { $ThreadsShared } else { $ThreadsFull }
    $args1 = $rest -replace '\{T\}', "$t"
    foreach ($m in [regex]::Matches($args1, '--(out|state|hess-out)\s+(\S+)')) {
      if ($m.Groups[1].Value -ne 'out' -or $m.Groups[2].Value -notmatch '\.(tsv|f32)$') {
        New-Item -ItemType Directory -Force $m.Groups[2].Value | Out-Null
      }
    }
    Log "$name start ($g, $t threads): $exe $args1"
    $sw = [Diagnostics.Stopwatch]::StartNew()
    $p = Start-Process -FilePath $exe -ArgumentList $args1 -RedirectStandardOutput "$runs\$name.out" -RedirectStandardError "$runs\$name.err" -NoNewWindow -PassThru
    try { $p.PriorityClass = 'BelowNormal' } catch {}
    $killed = $false
    while (-not $p.HasExited) {
      Start-Sleep 10
      if ((Gate) -eq 'closed' -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force; $killed = $true; break }
    }
    $p.WaitForExit()
    if ($killed) { Log "$name stopped for postpack after $([int]$sw.Elapsed.TotalSeconds) s; will rerun"; continue }
    $rc = $p.ExitCode
    Log ("{0} exit {1} after {2:N0} s" -f $name, $rc, $sw.Elapsed.TotalSeconds)
    if ($rc -eq 0) { New-Item -ItemType File -Force "$runs\$name.done" | Out-Null }
    else { Get-Content "$runs\$name.err" -Tail 5 | ForEach-Object { Log "  stderr: $_" } }
    break
  }
}
Log "queue $Queue finished"
