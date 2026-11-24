# NanoQuatTwo package v2, 2026-11-23: testing notes for JD

Everything in this package was built from the reviewed sources
on 2026-11-23 (conversion review and engine review applied).
The Windows exes were run under Wine on Elfie's bench before
shipping: the runner reproduced the Linux checksum exactly
(see test 1), the auditor matched, and the quantizer and
engine print their usage. They import only KERNEL32.dll and
msvcrt.dll; no MinGW runtime DLLs are needed.

New in v2: two probe tools (nq2-frameprobe, nq2-errprobe),
Gizmo's engineering handoff and our response to it, and a
baseline re-run of every anchor in these notes from the
shipped binaries late on 2026-11-23: all matched exactly,
including the frame probe producing identical output on
Linux and under Wine.

## Contents

- bin/win64: nq2-quantize.exe, nq2-run.exe, nq2-audit.exe,
  nq2-infer.exe, nq2-frameprobe.exe, nq2-errprobe.exe
  (static, ready to run)
- bin/linux64: the same six tools for Linux
- tools: the C sources (plus nq2-gui.py, a Tkinter front end
  that only calls the C runner)
- model-pack: sigma-full-b256-v2run.nq2 + .nq2book +
  .report.txt (Sigma-3B, global book, block 256; relative L2
  0.6922, a pipeline proof, not a quality artifact)
- examples: the original quadword VQ toy sources
- build.sh / build.bat: build all six tools from source
- Docs: README.md, WHAT-WE-CAN-DO.md, the three review files,
  the Greene research notes, Gizmo's notes, the preliminary
  report prepared for Gizmo, Gizmo's engineering handoff,
  and our response to it (RESPONSE-TO-GIZMO-2026-11-23.md).
- results-reports: the quantizer reports for the per-tensor
  staircase steps that have landed (blocks 32, 64, 128).

## Test 1: the runner on the packed model (no model download needed)

    bin\win64\nq2-run.exe model-pack\sigma-full-b256-v2run.nq2 model-pack\sigma-full-b256-v2run.nq2book

Expected:
- expanded checksum (FNV-1a 64): 7bd9557531df1ab7
- peak resident: 4476940 bytes
- compression: 0.1563 bits per weight (stream)

## Test 2: the auditor (no model download needed)

    bin\win64\nq2-audit.exe model-pack\sigma-full-b256-v2run.nq2 model-pack\sigma-full-b256-v2run.nq2book

Expected:
- COMPLETE description: 75146008 bytes, 0.15680 bits/weight
  (stream + book; pass bin\win64\nq2-infer.exe as a third
  argument to also charge the decoder, which raises the
  total by the exe's own size)
- H0 marginal: 6.4189 bits/index; H1: 5.7925
- codebook utilization: 256 of 256

## Test 3: the engine (needs the model directory)

The engine needs the raw Sigma shards in both modes: in
packed mode the tiny tensors (gamma bounds, noise schedule)
and all block tails are served from the raw shards by design.
Download nvidia/Nemotron-Labs-Sigma-3B from Hugging Face
(the two .safetensors files plus config.json) into a folder,
then:

    bin\win64\nq2-infer.exe --dir <model folder> --ids 1 --tokens 32 --steps 1 --guidance 1 --score-temp 1 --seed 42

    bin\win64\nq2-infer.exe --dir <model folder> --packed model-pack\sigma-full-b256-v2run.nq2 model-pack\sigma-full-b256-v2run.nq2book --ids 1 --tokens 32 --steps 1 --guidance 1 --score-temp 1 --seed 42

Notes: on Elfie's throttled bench each run took about 6
minutes; on your workstation expect far less. The pre-review
engine produced token IDs beginning 19254 60401 63533 on the
raw run; the reviewed binary may differ slightly at near
ties because the diffusion head's mean is now accumulated in
float64 (that difference is the fix working). The packed run
will not match the raw run at this pack's fidelity, and is
not expected to.

## Test 4: build from source

With MinGW gcc on PATH, run build.bat from the package root.
nq2-infer needs a POSIX-thread MinGW; the other five build
with either thread model. On Linux, build.sh. (MSYS2's
UCRT64 toolchain is POSIX-thread and builds all six.)

## Test 5: the frame probe (no model download needed)

    bin\win64\nq2-frameprobe.exe model-pack\sigma-full-b256-v2run.nq2

Expected headline lines:
- frame bytes: 59907056 of 74883852 stream bytes (80.0%);
  frame rate 0.12500 bits/weight
- min codes: 1796 distinct, H0 7.5178 bits;
  delta-coded min: H0 7.4991 bits
- scale codes: 3256 distinct, H0 8.5698 bits;
  scale log2-delta (1/64): H0 6.1412 bits

## Test 6: the error probe (needs the model directory)

    bin\win64\nq2-errprobe.exe model-pack\sigma-full-b256-v2run.nq2 model-pack\sigma-full-b256-v2run.nq2book <shard1.safetensors> <shard2.safetensors>

Expected headline lines:
- total relative L2: 0.6922 over 243 tensors
  (fallback weights unpacked: 34)
- replaid_embedding_matrix.matrix: rel L2 0.9922,
  err share 39.80%
- summary: top-10 tensors hold 54.6% of the squared error;
  median per-tensor rel L2 over 241 packed tensors: 0.9943

## Known open items (not hidden)

- The per-tensor staircase (block sizes 32 to 4096 on Sigma
  layers 0 to 3) is still running on Elfie's bench; blocks
  32, 64, and 128 have landed (reports in results-reports)
  and the rest, with the baseline block-size recommendation,
  follow separately. JD's current lean is 512.
- The response to Gizmo (RESPONSE-TO-GIZMO-2026-11-23.md)
  carries the frame findings and the per-tensor error
  breakdown in full.
- The engine is token-IDs in and out; there is no tokenizer
  or text chat yet.
- Numerical validation against NVIDIA's reference
  implementation has not been done; the engine is an
  independent C implementation of the published procedure.
