# parakeet-c: native Windows speech runtime for Nemo-family dictation

Engineering notes for `parakeet-c` (2026-11-26). Idea and shape: JD. User guide: [docs/DICTATION.md](../docs/DICTATION.md).

## What it is
A pure-C11 runtime (plus WASAPI COM calls) that runs NVIDIA **Parakeet TDT 0.6B v2**
fully on Windows. It doesn't use WSL, Linux libraries or Python anywhere in the speech path.
The chat GUI (Python/Tk) only spawns `parakeet-c.exe --stdio` and reads lines back.

| Stage | File | Notes |
|---|---|---|
| Mic capture | `src/mic_wasapi.c` | WASAPI shared mode, 16 kHz mono float via Windows SRC (AUTOCONVERTPCM). If the device refuses, falls back to the mix format plus our own windowed-sinc resampler. Polled from the main thread, so there is no capture thread. |
| Front end | `src/mel.c` | NeMo `AudioToMelSpectrogramPreprocessor`: preemph 0.97, 512-pt FFT, Hann(400, periodic=False), hop 160, center zero-pad, power, slaney mel 128, log(x+2^-24), per-feature norm with unbiased std+1e-5, T = n/160. Uses the checkpoint's own `fb` and `window` (our generated ones match to 4e-9 and 2e-7). |
| Encoder | `src/model.c` | FastConformer: dw_striding x8 subsampling (masked like `MaskedConvSequential`), 24 layers with macaron FF, rel-pos MHSA (pos_bias_u/v, rel_shift), conv module (GLU, depthwise k9, folded BN, swish). bf16 weights are used straight from the memory-mapped safetensors. |
| GEMM | `src/gemm.c` | AVX2+FMA 4x2 kernel that widens bf16 to f32 in registers. |
| Decoder | `src/model.c` | TDT greedy, matching NeMo `GreedyTDTInfer._greedy_decode` (durations 0..4, max_symbols 10, blank SOS). Also plain RNNT and CTC greedy for `parakeet-ctc-*`. |
| Tokenizer | `src/spm.c` | Reads the SentencePiece ModelProto directly. Detokenizes ▁ and byte pieces. |
| Threads | `src/util.c` | Lock-free fork/join pool: atomics only, created and joined around each decode (phase boundaries). **Zero mutexes.** |
| Streaming | `src/main.c` | Rolling segment. A partial re-decode runs every 0.4 s of new audio, adaptive so it never queues behind the CPU. A segment is finalized at a 0.7 s pause or at 12 s, cut at the quietest 100 ms. Energy VAD with an adaptive noise floor. Lone fillers ("uh", "um") and segments with under 0.2 s of speech are dropped. |

`accel/` is reserved for a CUDA encoder lane. It is not built yet because the CPU is already at about 0.06 RTF.

## Weights (reused, not downloaded)
The weights are not in this repository. Download them (or reuse an existing NeMo extract) into
`<models-dir>\parakeet-tdt-0.6b-v2\` and point `PARAKEET_MODEL` at it:
- `model_weights.safetensors` (1,253,859,392 B; encoder bf16, decoder/joint f32): the "fast" extract made from the .nemo on 2026-10-07.
- `*_tokenizer.model`, `*_tokenizer.vocab`, `*_vocab.txt`, `model_config.yaml`, `manifest.json`.
- `samples\*.wav` and `samples\ref\*.txt`: the WSL/NeMo pipeline's own transcripts, used as the accuracy reference.

Upstream: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2 (CC-BY-4.0).

## Build and run
```
build.bat                                   (MSYS2 UCRT64 gcc, -O3 -march=native, static)
bin\parakeet-tests.exe <models-dir>\parakeet-tdt-0.6b-v2
bin\parakeet-c.exe --timing file.wav        (WAV PCM 8/16/24/32 or float, any rate or channels)
bin\parakeet-c.exe --mic                    (console dictation: partials on stderr, finals on stdout, Enter stops)
bin\parakeet-c.exe --mic-list               (* marks the Windows default)
bin\parakeet-c.exe --mic-test 5 [--save]    (level meter; --save writes mic-test.wav)
bin\parakeet-c.exe --stream-test file.wav   (replays a WAV through the live path in real time)
bin\parakeet-c.exe --stdio                  (GUI protocol, below)
env: PARAKEET_MIC=<part of device name>, PARAKEET_THREADS, PARAKEET_HOP_MS, PARAKEET_PAUSE_MS, PARAKEET_MODEL
```

### stdio protocol
- **In:** `START`, `STOP`, `CANCEL`, `FILE <wav>`, `SIMFILE <wav>` (test: replays a WAV as the mic), `PING`, `QUIT`
- **Out:** `READY <info>`, `REC <mic>`, `LEVEL <0..1> <speaking>`, `PARTIAL <seg> <text>`, `FINAL <seg> <text>`,
  `STAT <seg> audio= mel= enc= dec= total= rtf= final=`, `WARN <msg>` (mic delivers digital silence), `DONE`, `ERR <msg>`, `RESULT <text>`

## GUI
- `gui/nemo_dictation.py` is the dictation client (a copy lives in `littlenemo/gui/`).
- `nemo_ui.py` carries a small guarded hook in `_build_composer`.
- How it works:
  - A big mic button and a "Click to dictate" label sit in the composer tool row.
  - Click to start and click to stop. You can also hold the button and let go (push-to-talk), or press Ctrl+Shift+Space.
  - While recording: a red stop button with a level ring, "Listening… m:ss" and a level meter.
  - Partial text streams into the composer in dim colour and firms up as phrases finalize. Nothing is sent; Enter sends as usual.
  - parakeet-c is pre-warmed 1.5 s after the GUI opens, so the first click is instant.
  - parakeet-c exits when the GUI's pipe closes.

## Results (12th-gen Core i5-12450H, 8 threads, CPU only)
| Test | Result |
|---|---|
| Unit tests | 17 PASS / 0 FAIL (FFT, GEMM f32/bf16 at 1 and 8 threads, 20k pool jobs, resampler + anti-alias, WAV, mel, SPM, checkpoint fb/window/dims/vocab) |
| LibriSpeech 2086-149220-0033 (7.43 s) | exact match to the WSL/NeMo reference; 0.45 s warm (RTF 0.06) |
| espeak-test (7.79 s) | exact match; 0.41–0.45 s warm |
| Streaming replay | partial decodes 0.05–0.45 s; final text 0.15–0.5 s after the end of speech; finals exact |
| GUI end to end (SIMFILE) | composer text equals the reference on both samples; screenshots in `shots\` |
| Memory | about 59 MB private; the 1.2 GB of weights are a shared, file-backed mapping (pageable) |
