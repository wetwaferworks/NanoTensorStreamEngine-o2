# bench-cuda-io.ps1 - tokens-in / tokens-out bench for the Little Nemo CUDA lane.
# Feeds raw (non-chat) prompts of several lengths through --serve-stdio,
# RESET between prompts, and prints prompt tokens, prefill ms and tok/s,
# TTFT and decode tok/s from the engine's own STATS lines.
# Usage: powershell -File accel\bench\bench-cuda-io.ps1 -Exe <engine.exe> -Ckpt <dir> -Pack <nq2> -Out <log> [-Expand never|auto|always] [-MaxNew 32] [-Extra '...']
param(
  [Parameter(Mandatory)] [string]$Exe,
  [Parameter(Mandatory)] [string]$Ckpt,
  [Parameter(Mandatory)] [string]$Pack,
  [Parameter(Mandatory)] [string]$Out,
  [string]$Expand = 'never',
  [int]$MaxNew = 32,
  [string]$Words = '8,48,96,192,384',
  [int]$Reps = 1,
  [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
$para = 'The river ran past the old mill where the farmers brought their grain each autumn, and the miller kept a ledger of every sack, every price and every promise made over the long winter months while the children played along the frozen banks and the wheel stood still under a coat of ice.'
$pw = $para -split ' '
$lines = New-Object System.Collections.Generic.List[string]
for ($r = 0; $r -lt $Reps; $r++) {
  foreach ($n in ($Words -split ',' | ForEach-Object { [int]$_ })) {
    $w = for ($i = 0; $i -lt $n; $i++) { $pw[$i % $pw.Count] }
    $lines.Add('PROMPT ' + ($w -join ' '))
    $lines.Add('RESET')
  }
}
$lines.Add('QUIT')
$inFile = [IO.Path]::ChangeExtension($Out, '.in.txt')
[IO.File]::WriteAllLines($inFile, $lines)
$tok = Join-Path $Ckpt 'tokenizer.json'
$args = @($Ckpt, $tok, '--max-new', "$MaxNew", '--serve-stdio', '--accel', 'cuda', '--cuda-pack', $Pack, '--expand', $Expand)
if ($Extra) { $args += ($Extra -split ' ') }
$errFile = [IO.Path]::ChangeExtension($Out, '.err.txt')
$p = Start-Process -FilePath $Exe -ArgumentList $args -RedirectStandardInput $inFile -RedirectStandardOutput $Out -RedirectStandardError $errFile -NoNewWindow -PassThru
$peak = 0
while (-not $p.HasExited) {
  Start-Sleep -Milliseconds 250
  $q = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
  if ($q) { $v = [int]($q | Select-Object -First 1); if ($v -gt $peak) { $peak = $v } }
}
Get-Content $Out | Where-Object { $_ -like 'STATS*' } | ForEach-Object {
  $h = @{}; foreach ($kv in ($_ -split ' ')) { if ($kv -match '^(\w+)=(.*)$') { $h[$matches[1]] = $matches[2] } }
  '{0,5} in  prefill {1,7} ms {2,8} tok/s   ttft {3,7} ms   out {4,3} tok {5,8} tok/s' -f $h.prompt_tokens, $h.prompt_ms, $h.prompt_tps, $h.ttft_ms, $h.tokens, $h.gen_tps
}
"peak GPU memory used (whole card, nvidia-smi): $peak MiB"
