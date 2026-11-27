# test_sn_engine_pending.ps1
# Drive sn-engine with a long PROMPT, a short command queued during
# generation, and another getline. Found in Archie's round 7 review.
# Usage: test_sn_engine_pending.ps1 -Engine <sn-engine.exe> [-Prefix testdata\tiny\tiny]
param(
  [Parameter(Mandatory=$true)][string]$Engine,
  [string]$Prefix = "testdata\tiny\tiny",
  [string]$Root = ""
)
$ErrorActionPreference = 'Stop'
if ($Root) { Set-Location $Root }
(Get-Process -Id $PID).PriorityClass = 'BelowNormal'
if (-not (Test-Path $Engine)) { Write-Output "FAIL engine not found: $Engine"; exit 1 }
$shapes = (Join-Path (Split-Path $Prefix -Parent) 'shapes.tsv')
$config = (Join-Path (Split-Path $Prefix -Parent) 'config.json')
$tok = (Join-Path (Split-Path $Prefix -Parent) 'tokenizer.json')
foreach ($p in @("$Prefix.nq2", $shapes, $config, $tok)) {
  if (-not (Test-Path $p)) { Write-Output "FAIL missing $p"; exit 1 }
}

$long1 = 'PROMPT ' + ('A' * 4000)
$short = 'RESET'
$long2 = 'PROMPT ' + ('B' * 2000)

$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = (Resolve-Path $Engine).Path
$psi.Arguments = "--pack `"$Prefix`" --shapes `"$shapes`" --config `"$config`" --tokenizer `"$tok`" --max-new 8 --threads 1 --below-normal --cache-mb 0 --no-spec"
$psi.UseShellExecute = $false
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.CreateNoWindow = $true
$p = New-Object System.Diagnostics.Process
$p.StartInfo = $psi
$p.Start() | Out-Null
try { $p.PriorityClass = 'BelowNormal' } catch {}

# Wait for READY, then feed the three-line sequence plus QUIT.
$deadline = [datetime]::UtcNow.AddSeconds(60)
$ready = $false
$out = New-Object System.Text.StringBuilder
while ([datetime]::UtcNow -lt $deadline) {
  if ($p.StandardOutput.Peek() -ge 0 -or $p.HasExited) {
    $line = $p.StandardOutput.ReadLine()
    if ($null -eq $line) { break }
    [void]$out.AppendLine($line)
    if ($line.StartsWith('READY')) { $ready = $true; break }
  } else {
    Start-Sleep -Milliseconds 20
  }
}
if (-not $ready) {
  if (-not $p.HasExited) { $p.Kill() }
  Write-Output "FAIL engine did not become READY"
  Write-Output $out.ToString()
  Write-Output $p.StandardError.ReadToEnd()
  exit 1
}

$p.StandardInput.WriteLine($long1)
$p.StandardInput.WriteLine($short)
$p.StandardInput.WriteLine($long2)
$p.StandardInput.WriteLine('QUIT')
$p.StandardInput.Close()

if (-not $p.WaitForExit(120000)) {
  $p.Kill()
  Write-Output "FAIL engine timed out"
  exit 1
}
$rest = $p.StandardOutput.ReadToEnd()
[void]$out.Append($rest)
$err = $p.StandardError.ReadToEnd()
$text = $out.ToString()
$rc = $p.ExitCode
Write-Output ("engine exit {0}" -f $rc)
if ($text -match 'READY') { Write-Output 'PASS engine became READY' } else { Write-Output 'FAIL no READY'; $rc = 1 }
if ($text -match 'DONE') { Write-Output 'PASS engine produced DONE' } else { Write-Output 'FAIL no DONE'; $rc = 1 }
# A heap overflow typically exits non-zero or never reaches QUIT cleanly.
if ($p.ExitCode -eq 0) { Write-Output 'PASS engine exited 0 after the long/short/long command sequence' }
else { Write-Output ("FAIL engine exited {0}" -f $p.ExitCode); if ($err) { Write-Output $err } ; $rc = 1 }
if ($rc -ne 0 -and $err) { Write-Output $err }
exit $rc
