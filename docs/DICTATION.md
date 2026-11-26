# Dictation: parakeet-c

Press a mic button in the chat window and talk; your words appear in the composer as you speak. The speech recognizer is `parakeet-c`, a native Windows runtime for NVIDIA's Parakeet TDT 0.6B v2, written in pure C11 plus WASAPI calls: no Python, WSL or Linux library anywhere in the speech path. The GUI only spawns `parakeet-c.exe --stdio` and reads lines back. Source: `parakeet/`. Idea and shape: JD. Implementation: Cortana, with the other assistants.

## What you need

- A Windows machine with a microphone and MSYS2 UCRT64 gcc to build.
- Your own copy of the Parakeet weights (not in this repository): the model folder holds `model_weights.safetensors` (encoder bf16, decoder and joint f32), the `*_tokenizer.model`, `*_tokenizer.vocab`, `*_vocab.txt`, `model_config.yaml` and `manifest.json`. Upstream: <https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2> (CC-BY-4.0).
- `PARAKEET_MODEL` pointing at that folder (below).

## Build

```
cd parakeet
build.bat          # -O3 -march=native -mavx2 -mfma, static; builds bin\parakeet-c.exe and bin\parakeet-tests.exe
```

## Quick start

```
set PARAKEET_MODEL=<path to parakeet-tdt-0.6b-v2>
bin\parakeet-c.exe --mic-list          :: pick your microphone (* marks the Windows default)
bin\parakeet-c.exe --mic-test 5        :: 5 seconds of level meter, no model needed
bin\parakeet-c.exe --timing file.wav   :: transcribe a file
bin\parakeet-c.exe --mic               :: live dictation in the console
```

In the chat window: click the mic, or press **Ctrl+Shift+Space**; click again (or press the shortcut again) to stop.

## Model location: PARAKEET_MODEL

`parakeet-c.exe` finds the model folder in this order: `--model DIR`, the `PARAKEET_MODEL` environment variable, then `models\parakeet-tdt-0.6b-v2` and `..\models\parakeet-tdt-0.6b-v2` relative to the working directory (the first one that contains `model_weights.safetensors`). The GUI passes `--model` itself: it reads `PARAKEET_MODEL` from its environment, and if that is not set it reads the Windows **user** environment (`HKEY_CURRENT_USER\Environment`), so a shortcut that was started before you set the variable still works. To set it for good: `setx PARAKEET_MODEL "<path>"` (new processes only).

## Command line

| Flag | Meaning |
|---|---|
| `--model DIR` | model folder |
| `--threads N` | worker threads (default: `PARAKEET_THREADS`, else the machine's choice) |
| `--timing` | print per-stage timing for each file |
| `--bench N` | repeat the decode N times and report |
| `file.wav [...]` | transcribe WAV files (PCM 8/16/24/32 or float, any rate or channel count) |
| `--mic [--seconds N]` | live dictation in the console: partials on stderr, finals on stdout, Enter stops |
| `--mic-list` | list microphones |
| `--mic-test [N] [--save]` | level meter for N seconds, no model needed; `--save` writes `mic-test.wav` |
| `--stream-test file.wav` | replay a WAV through the live dictation path in real time |
| `--stdio` | the GUI line protocol (below) |
| `--help`, `-h` | usage |

### Environment variables

| Variable | Meaning |
|---|---|
| `PARAKEET_MODEL` | model folder (above) |
| `PARAKEET_MIC` | part of a device name, to pick a microphone other than the default |
| `PARAKEET_MIC_ROLE=communications` | use the Windows "communications" default device instead of the console default |
| `PARAKEET_THREADS` | worker threads when `--threads` is not given |
| `PARAKEET_HOP_MS` | how often a partial is re-decoded (default 400 ms) |
| `PARAKEET_PAUSE_MS` | pause length that ends a segment (default 700 ms) |
| `PARAKEET_EXE` | (GUI) full path to `parakeet-c.exe` |

The GUI looks for the executable in this order: `PARAKEET_EXE`, `parakeet/bin/parakeet-c.exe` beside the `littlenemo` folder, `littlenemo/bin/parakeet-c.exe`, then `parakeet/parakeet-c.exe` beside the GUI sources.

## Microphone selection

Capture is WASAPI shared mode: 16 kHz mono float through Windows' own sample-rate converter; if the device refuses, the runtime falls back to the mix format and its own windowed-sinc resampler. It is polled from the main thread, so there is no capture thread. Run `--mic-list`, then either make that device the Windows default or set `PARAKEET_MIC` to part of its name. If the device delivers digital silence (some noise-cancelling virtual microphones do), the runtime sends `WARN` and the GUI says "mic silent? check NVIDIA Broadcast / PARAKEET_MIC".

## Streaming partials

The runtime keeps a rolling segment. A partial re-decode runs every 0.4 s of new audio, adaptively so it never queues behind the CPU. A segment is finalized at a 0.7 s pause or at 12 s, cut at the quietest 100 ms. An energy detector with an adaptive noise floor finds speech; lone fillers ("uh", "um") and segments with under 0.2 s of speech are dropped.

## The stdio protocol

One line per message.

| Direction | Message | Meaning |
|---|---|---|
| in | `START` / `STOP` / `CANCEL` | begin, end (and finalize), or abandon a recording |
| in | `FILE <wav>` | transcribe a file |
| in | `SIMFILE <wav>` | test: replay a WAV as if it were the microphone |
| in | `PING` | liveness check (answered `PONG`) |
| in | `QUIT` | exit |
| out | `READY <info>` | model loaded |
| out | `REC <mic>` | recording started on this device |
| out | `LEVEL <0..1> <speaking>` | input level |
| out | `PARTIAL <seg> <text>` | provisional text for a segment |
| out | `FINAL <seg> <text>` | finished text for a segment |
| out | `STAT <seg> audio= mel= enc= dec= total= rtf= final=` | timing for a segment |
| out | `WARN <msg>` | for example, the mic delivers digital silence |
| out | `DONE` | the recording is complete |
| out | `ERR <msg>` | an error |
| out | `RESULT <text>` | the transcript of a `FILE` request |

## In the chat window

A big mic button and a "Click to dictate" label sit in the composer tool row. Click to start and click to stop; you can also hold the button and let go (push-to-talk), or press Ctrl+Shift+Space. While recording there is a red stop button with a level ring, "Listening... m:ss" and a level meter. Partial text streams into the composer in a dim color and firms up as each phrase is finalized. **Nothing is sent**: the text stays in the composer for review, and Enter sends as usual. `parakeet-c` is pre-warmed 1.5 s after the window opens so the first click is instant, and it exits when the GUI's pipe closes. If `parakeet-c.exe` is not found the label says "mic: parakeet-c.exe not found".

## How it works

| Stage | File | Notes |
|---|---|---|
| Front end | `src/mel.c` | the NeMo mel preprocessor (pre-emphasis 0.97, 512-point FFT, Hann 400, hop 160, 128 slaney mel bins, log, per-feature normalization), using the checkpoint's own filterbank and window |
| Encoder | `src/model.c` | FastConformer: 8x depthwise-striding subsampling, 24 layers with macaron feed-forward, relative-position attention and a convolution module; bf16 weights used straight from the memory-mapped safetensors |
| GEMM | `src/gemm.c` | AVX2+FMA 4x2 kernel that widens bf16 to f32 in registers |
| Decoder | `src/model.c` | TDT greedy, matching NeMo's `GreedyTDTInfer._greedy_decode` (durations 0 to 4, max 10 symbols); plain RNNT and CTC greedy for `parakeet-ctc-*` |
| Tokenizer | `src/spm.c` | reads the SentencePiece model directly |
| Threads | `src/util.c` | lock-free fork/join pool, atomics only, **zero mutexes** |

`accel/` is reserved for a CUDA encoder lane and is not built, because the CPU is already at about 0.06 real-time factor.

## Measured

On a 12th-gen Core i5-12450H, 8 threads, CPU only:

| Test | Result |
|---|---|
| Unit tests (`parakeet-tests.exe`) | 17 pass, 0 fail: FFT, GEMM f32 and bf16 at 1 and 8 threads, 20k pool jobs, resampler and anti-alias, WAV, mel, SPM, checkpoint filterbank, window, dims and vocabulary |
| LibriSpeech 2086-149220-0033 (7.43 s) | exact match to the NeMo reference transcript; 0.45 s warm (RTF 0.06) |
| a synthesized-speech clip (7.79 s) | exact match; 0.41 to 0.45 s warm |
| Streaming replay | partial decodes 0.05 to 0.45 s; final text 0.15 to 0.5 s after the end of speech; finals exact |
| GUI end to end (`SIMFILE`) | composer text equals the reference on both samples |
| Memory | about 59 MB private; the 1.2 GB of weights are a shared, file-backed mapping |

## Tests

`parakeet/tests/`: `test_units.c` (built as `bin\parakeet-tests.exe`, takes the model folder), `stdio_smoke.py` (drives `--stdio`: READY, PING, FILE, START/STOP, QUIT) and `gui_dictation_test.py` (launches the chat window with a fixture engine, replays a reference WAV as the microphone, checks the composer text). See [TESTING.md](TESTING.md).

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| "mic: parakeet-c.exe not found" | build it (`parakeet\build.bat`) or set `PARAKEET_EXE` |
| error about the model or weights | set `PARAKEET_MODEL` to the folder holding `model_weights.safetensors`, or pass `--model` |
| the mic button does nothing right after `setx` | environment changes reach new processes only; restart the chat window (the GUI also reads the Windows user environment) |
| recording starts but no words | `--mic-test 5` to see the level; `--mic-list`, then `PARAKEET_MIC`; try `PARAKEET_MIC_ROLE=communications` |
| "mic silent?" | the device delivers digital silence; check NVIDIA Broadcast or another virtual device, or choose the physical microphone |
| words arrive late | partials re-decode every 0.4 s and a segment finalizes after a 0.7 s pause; lower `PARAKEET_HOP_MS` and `PARAKEET_PAUSE_MS` at some CPU cost |