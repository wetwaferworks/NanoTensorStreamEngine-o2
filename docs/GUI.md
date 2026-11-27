# The NTSE Chat GUI

A desktop chat window for the Little Nemo engine (and, when it exists, SuperNemo):
the conversation, the model's thinking on one live line above each reply, and a
stats panel that shows the machine working. The front end is Python and Tk; every
number it shows comes from the C engine over the line protocol, or is labeled as
measured by the GUI. It does no model work. Source: `littlenemo/gui/`. GUI art,
themes and mascots: CloudAtlas. Implementation: Elfie, Cortana and the other
assistants, with JD directing the shape.

- [Launching](#launching) and [command line](#command-line)
- [Configuration (littlenemo.ini)](#configuration-littlenemoini)
- [Models](#models)
- [The conversation](#the-conversation): input, keyboard shortcuts, attachments, thinking line
- [Stats panel](#stats-panel)
- [Themes](#themes), [mascots and icon packs](#mascots-and-icon-packs), [shape customizer](#shape-customizer)
- [Settings dialog](#settings-dialog)
- [C scaler](#c-scaler), [DPI and resizing](#dpi-and-resizing)
- [Protocol v3 message reference](#protocol-v3-message-reference)
- [Modules](#modules), [testing](#testing), [troubleshooting](#troubleshooting)

## Launching

Requirements: Python 3 with Tk (the standard Windows installer includes it), and
Pillow for attachment inspection. Dictation additionally needs the Parakeet engine
([DICTATION.md](DICTATION.md)). The scaler library must be built once
(see [C scaler](#c-scaler)); without it the GUI uses the nearest prebuilt asset size.

```
cd littlenemo
build.bat                      # builds nemo-engine and the test suites into bin/
python gui/nemo_chat.py        # reads littlenemo.ini; no arguments needed
```

For the chat-templated engine (`engine_gui`, built by hand; see
[LITTLE-NEMO-ENGINE.md](LITTLE-NEMO-ENGINE.md)), point the GUI at it:

```
python gui/nemo_chat.py --engine "bin/engine_gui.exe <checkpoint_dir> <tokenizer.json> --chat --think --mode max"
```

The INI-built engine command is `engine <checkpoint> <tokenizer>` plus `--max-new`,
`--mode`, `--threads`, `--simd`, `--max-pos`, so `--chat` and `--think` can only be
supplied through `--engine`.

For a desktop launch with no console window, run `pythonw gui/nemo_chat.pyw` (a one-line launcher
that writes anything that would have gone to the console to `%LOCALAPPDATA%\NemoChat\gui.log`).
The engine, the converter and every helper program the GUI starts are also spawned without a
console window unless **Show console windows** is on ([Settings dialog](#settings-dialog)).

With no engine and no INI the window opens with every model shown as unavailable and
sends nothing. To see every GUI path without a model, use the test fixture (below).

## Command line

| Flag | Meaning |
|---|---|
| `--config PATH` | the INI to read and write; default search is `./littlenemo.ini`, beside the script, then the script's parent directory |
| `--engine CMD` | the Little Nemo engine command line (overrides `[paths]`) |
| `--host H` and `--port P` | connect to an engine already running with `--serve PORT` (overrides `[remote]`); see the warning under the configuration table |
| `--model supernemo\|littlenemo` | which model is selected at startup |
| `--supernemo-engine CMD` | the SuperNemo engine command line (overrides `[supernemo] engine`) |
| `--supernemo-host H` and `--supernemo-port P` | a remote SuperNemo engine |
| `--theme ID` | theme for this session only (not saved): `pixel`, `lattice`, `stream`, `handshake`, `owl` |
| `--show-console` | start the engine and helpers in a visible console window for this session (default hidden; see **Show console windows**) |
| `--demo NAME` | scripted walk-through: `tour` (with screenshots), `smoke` (the same checks, no shots), `unavailable` (SuperNemo with no engine configured), `composer` (the input bar and dialogs, with screenshots) |
| `--shots DIR` | where `--demo tour` writes screenshots |

Engine options (`--mode`, `--threads`, `--simd`, `--max-new`, `--max-pos`) are not GUI flags: set them in `[engine]` in the INI, or put them in your own `--engine` command line.

### The test fixture

`littlenemo/tests/fake_engine.py` is a fixture, not a model. It speaks protocol v2 and
every v3 extension so the whole GUI can be exercised without weights. Its READY line
carries `fixture=1`, the GUI shows a **TEST FIXTURE** badge, and its replies say they
are not model output, so a screenshot can never pass a fixture reply off as model
output.

```
python gui/nemo_chat.py --model supernemo ^
  --supernemo-engine "python ../tests/fake_engine.py --stdio" ^
  --engine "python ../tests/fake_engine.py --stdio --model littlenemo" ^
  --demo tour --shots ./shots
```

Fixture options: `--model supernemo|littlenemo` (SuperNemo: v3, `media=image,video`,
MoE telemetry; Little Nemo: text-only READY), `--delay S` (seconds between streamed
tokens, default 0.03), `--v2` (no v3 lines at all), `--accel LANE` (reported in `HW
accel=`, default `cpu`), `--think N` (thinking tokens to stream), `--stdio` or `--port PORT`.

## Configuration (littlenemo.ini)

The INI shipped in `littlenemo/` is the template. Relative paths resolve against the
INI's own directory. Command-line flags override the file. The GUI writes settings
back to the same file and preserves its comments.

| Section | Keys | Notes |
|---|---|---|
| `[paths]` | `engine`, `checkpoint`, `tokenizer`, `cuda_pack` | the first three are needed for the GUI to build the engine command; `cuda_pack` is the `.nq2` pack for a CUDA engine (`--cuda-pack`) |
| `[cuda]` | `accel`, `expand`, `full_path`, `host_threads`, `tok_overlap`, `lm_head`, `prefill_chunk`, `dtc_margin`, `dtc_entropy`, `dtc_prompt_tokens`, `dtc_complexity`, `dtc_longform` | options of the CUDA engine, one per engine flag ([ACCEL-CUDA.md](ACCEL-CUDA.md#flags)); blank or `default` passes nothing, so the engine's own (fastest) default applies; sent only to an engine whose file name contains `cuda` |
| `[gui]` | `show_console` | `true` shows the engine and helper console windows; default `false` (hidden) |
| `[dictation]` | `parakeet_exe`, `parakeet_model` | where to find `parakeet-c.exe` and the Parakeet model; blank searches ([DICTATION.md](DICTATION.md#environment-variables)) |
| `[engine]` | `mode` (default `max`), `threads` (0 = let the mode decide), `simd` (true/false), `max_new` (256), `max_pos` (4096) | see [LITTLE-NEMO-ENGINE.md](LITTLE-NEMO-ENGINE.md#engine-flags) |
| `[remote]` | `host`, `port` | to talk to an engine on another machine; leave `[paths]` empty |
| `[supernemo]` | `engine`, `host`, `port` | empty until a SuperNemo engine exists |
| `[appearance]` | `theme` | theme id |
| `[model.<key>]` | `icon`, `shape`, `fill`, `outline`, `face`, `size`, `motion`, `speed` | per-model look; user-defined models also take `label`, `full`, `engine`, `host`, `port`, `inputs` |

**Remote engines and the network.** The engine protocol has no authentication: whoever reaches the port can send PROMPT, RESET and QUIT. `nemo-engine --serve PORT` therefore listens on 127.0.0.1 only, which is enough for a GUI on the same machine or one that reaches it through an SSH tunnel (`ssh -L PORT:127.0.0.1:PORT host`, then point `[remote]` at 127.0.0.1). To accept a GUI on another machine directly, start the engine with `--serve PORT --serve-any`; it then listens on every interface and prints a warning. Do that only on a network you trust, behind a firewall.

## Models

The sidebar lists the models and each has its own conversation state and look.

| Model | Status |
|---|---|
| **Little Nemo** | works: the C engine on Nemotron-Nano-9B-v2, text only; its READY carries no `media` key, so attachments are refused before anything is sent |
| **SuperNemo** | the target is Nemotron 3.5 Super VL 120B-A12B (512-expert MoE, vision tower). NTSE has no engine path for it yet, so with no `[supernemo]` engine the window says the engine is not available yet and sends nothing. The GUI side (image and video attachments, MoE telemetry) is built and tested against the fixture |
| user-defined | add a `[model.<key>]` section with `label`, `engine` (or `host` and `port`) and `inputs` and the model appears in the sidebar |

Selecting a model: click its row. Double-clicking a row opens the Appearance dialog
for it.

## The conversation

### Sending and keyboard shortcuts

| Key | Action |
|---|---|
| Enter | send the message |
| Shift+Enter | newline in the message (the bar wraps at words, grows to six lines, then scrolls) |
| Ctrl+V | paste; if the clipboard holds an image it is attached |
| Ctrl+Shift+Space | start or stop dictation (needs [Parakeet](DICTATION.md)) |
| mouse wheel | scrolls the transcript, the theme list and the stats panel under the pointer |

The message box is a multi-line text field. Its placeholder reads `Message <model name>...`, sits on
the first line where the cursor starts, and follows the selected model. Pasting with Ctrl+V resizes
the bar like typing does.

Buttons: send, stop (ends generation early with `STOP`), attach image, attach video,
attach file, new chat (`RESET`), copy and clear on messages, settings and model
switch. Every button has a plated and a glyph form per theme. Sidebar: **Stats
panel**, **Appearance...**, **Settings...**, **Restart engine**, **About and credits...**.

### Attachments

A model that advertises `media` in READY accepts images and video. Bounds are
checked in the GUI before anything is sent, and the engine must check them again:

| Limit | Value |
|---|---|
| attachments per turn | 8 |
| image | up to 64 MiB, 16384 px per side, 100 MP |
| video | up to 4 GiB, at most 32 frames sampled |
| formats | PNG, JPEG, GIF, BMP, WebP; MP4/M4V/MOV, WebM/MKV, AVI; animated GIF/WebP (extension and leading bytes must agree) |

Video frames are extracted with `ffmpeg` if it is on PATH, otherwise Pillow handles
animated GIF and WebP. Each attachment is announced with `MEDIA` (and `FRAME` lines
for video); the engine answers `MEDIA_OK` with its context-token cost or `MEDIA_ERR`
with a reason and the turn goes on without it. `RESET` drops pending attachments. The
`path` in `MEDIA` is absolute on the GUI's machine, so a remote engine needs a shared
path. The Stats panel lists attachments for the current turn.

### The thinking line

Each reply carries one status line above the answer showing what the model is thinking
about right now, streamed from `THINK` messages and collapsible to show the whole
trace. The line carries the active model's mascot (or the shape you picked) and
changes with the engine's events:

| State | Triggered by |
|---|---|
| idle | no turn in flight |
| thinking | `THINK` lines, or after Send before the first `TOK` |
| streaming | `TOK` lines (the answer is landing) |
| expanding | `EXPAND` with `state=begin` |
| dtc | `DTC` with `state=begin` |
| error | `ERROR`, or a failed turn; the title reads "Stopped by an error" |
| done | `DONE` |

`state=end` returns to thinking or streaming. Animation uses Tk `after()` only and
stops when the window is hidden or after the turn settles, so an idle chat costs no CPU.

## Stats panel

A live panel on the right (toggle with the **Stats panel** button). When the window
is narrower than its minimum size allows (Windows Snap, a small or rescaled display),
the panel steps aside so the chat column keeps its room, and it comes back when the
window is wide enough again; the **Stats panel** choice is not changed. In the chat
bar the send button always keeps its place: on a narrow bar the key hint goes first,
then the "Click to dictate" text (the mic icon stays), and mic messages wrap. Every value comes
from the engine's messages; when the engine does not send one, the panel shows a dash,
or a value the GUI measured itself and labels `gui`. Sections:

| Section | Source |
|---|---|
| Throughput | `TICK`, `STATS` (tokens, tokens/s, ms per token, elapsed, time to first token, prompt tokens, prompt tokens/s) |
| Hardware | `HW` (cpu, cores, threads, ram, gpu, vram, driver, igpu, accel, os), else what the GUI probed from its own machine |
| Now | `LAYER`, `TICK` (current layer and type, next layer, active, staged, mode) |
| Tensor Handshake | `HS` (pass, ordinal, hits, misses, stall ms, dtc, dtc ms, staged, more counters) |
| Memory | `MEM` (RAM, peak, VRAM, VRAM peak; VRAM is sampled by the GUI on a worker thread when the engine does not send it) |
| Attachments this turn | the GUI's own list with `MEDIA_OK` costs |
| Last turn | `STATS` |
| Other engine stats | any `STAT key=value` line, shown as is |
| Engine READY | the READY line as received |

`gbs` in `TICK` is an estimate (weight bytes touched this token over compute time) and
is labeled as one.

## Themes

Five themes ship, from CloudAtlas's v1 theme pack (`gui/theme-pack/`, loaded
directly from `theme.json`). JD approved all five. Themes apply live without restarting
the window or the engine.

| id | Name | Subtitle | Pack folder |
|---|---|---|---|
| `pixel` (default) | Pixel Nemo | Retro Pixel | `themes/5-pixel-nemo/` |
| `lattice` | Lattice | Tensor Crystal | `themes/1-lattice/` |
| `stream` | Stream | Bitstream | `themes/2-stream/` |
| `handshake` | o.O Handshake | | `themes/3-handshake-oO/` |
| `owl` | Sage Owl | | `themes/4-sage-owl/` |

Each theme supplies the window icon (`.ico`, 16 to 256 px), tray icons, a picker banner
(1200x300), ten button actions in plated and glyph forms (`send`, `stop`,
`attach-image`, `attach-video`, `attach-file`, `new-chat`, `settings`, `model-switch`,
`copy`, `clear`), a 16x16x12 thinking strip, status dots (`ready`, `thinking`,
`generating`, `expanding`, `dense-compute`, `error`) and background tiles. Invalid
values are skipped so a typo cannot crash the UI. If the pack is missing, the picker
falls back to two built-in themes (Night and Ember, kept in `gui/themes/`). Pick a
theme in **Appearance...**; the choice is saved as `[appearance] theme`.

**Making your own theme.** The official pack is the CloudAtlas `theme.json` format
(colors, font aliases that map to installed families and never download a font, corner
radius, status-dot shape and asset paths; see [THEMES.md](../littlenemo/gui/THEMES.md)).
For your own, the simpler route is a single JSON file named `<id>.json` in a `themes/`
folder beside `littlenemo.ini` (or in `gui/themes/`): one object with `id`, `name`,
`palette` (hex `#RGB` or `#RRGGBB`, anything else is ignored), `fonts` (family and sizes 6
to 48), `spacing` (integers 4 to 800, scaled for DPI) and `icons`. A drop-in cannot
override an official id. The complete palette, with every key, is in THEMES.md; keys you
leave out are inherited, so a typo cannot leave the window without colors. Then pick it in
**Appearance...** or set `theme = <id>`.

```
{
  "id": "reef",
  "name": "Reef",
  "palette": { "bg": "#0E0E10", "accent": "#5EEAD4", "accent2": "#FF9B42" },
  "icons": "default"
}
```

`gui/theme-pack/scripts/` is build tooling only and is not shipped.

## Mascots and icon packs

A mascot is a small animated character on the thinking line and in the model list,
with seven states (idle, thinking, streaming, expanding, dtc, error, done). When a model
has no `icon=` of its own, the mascot that matches the active theme is used.

Official mascots (`ntse-mascot/1`, CloudAtlas) live in `gui/icons/`, one folder per theme:
`pixel-nemo` (default), `lattice-cube`, `stream-critter`, `handshake-oO` and `sage-owl`.
Each folder has a `manifest.json` and `<state>-32.png` / `<state>-64.png` horizontal
strips; the 64 px art is an exact 2x of the 32 px art.

**Make your own icon pack.** The simplest form is a folder with a `manifest.json` and
numbered frames or one horizontal strip:

```
{
  "name": "My Critter",
  "states": {
    "idle":      { "frames": [0],          "fps": 1,  "loop": false },
    "thinking":  { "frames": [1, 2],       "fps": 6,  "loop": true,  "motion": "bob" },
    "streaming": { "frames": [3, 4, 5],    "fps": 10, "loop": true,  "motion": "wobble" },
    "error":     { "frames": [6],          "fps": 1,  "loop": false },
    "done":      { "frames": [7],          "fps": 1,  "loop": false }
  }
}
```

Rules, from [ICONS.md](../littlenemo/gui/ICONS.md): `frames` are 0-based indices into the
strip and must land inside the pack (an out-of-range index rejects the pack and the GUI
falls back); `fps` is in (0, 60] and playback is capped at 20 fps; `loop` defaults to
true (`error` and `done` should be false); `motion` is optional, one of `flip`, `bob`,
`wobble`, `spin`, `pulse` (omit it if you painted the motion into the frames); a state you
leave out is filled from a default. Use a transparent background; the default palette is
purple `#A970FF`, orange `#FF9B42`, ink `#1A1228` and eye-white `#F7F4FF`; keep silhouettes
distinct at 32 px; do not copy anyone else's brand marks. Put the folder in `icons/`
beside `littlenemo.ini` (searched after the bundled icons) or pick it with **Appearance...
Browse**, and the choice is saved as `[model.<key>] icon = ...`. The `icon` value may also
be an absolute path, `builtin:<shape>` or `theme-strip:<theme id>`.

## Shape customizer

If you do not want to paint art, **Appearance...** offers a procedural "Thinking shape"
for each model, drawn by the GUI with no files at all:

| Setting | Values |
|---|---|
| shape | `blob`, `orb`, `cube`, `star`, `ring`, `droplet` |
| fill and outline | any hex color (chips offer quick picks) |
| face | on or off |
| size | 32 or 64 (anything 48 and above snaps to 64) |
| motion | `flip`, `bob`, `wobble`, `spin`, `pulse` |
| speed | 0.25 to 3.0 (1.0 default) |

Defaults: SuperNemo is a purple blob with an orange outline that bobs; Little Nemo is
an orange droplet with a purple outline that flips. Bad values fall back to the model's
built-in personality. The result is saved in `[model.<key>]`.

## Settings dialog

**Settings...** edits `littlenemo.ini`: the engine, checkpoint and tokenizer paths, mode,
threads, SIMD, `max_new`, `max_pos`, the CUDA options, the console toggle, the voice-input
paths and the remote host and port. The dialog scrolls (the Apply and Cancel bar stays in view),
validates the values and then **restarts the engine** so they take effect; **Restart engine** does
the same without changing anything.

**CUDA engine** (used only when the engine is a `nemo-engine-cuda` build). Blank or `default`
sends no flag, so the engine's own default applies; every default is the fastest setting.
`exact`, `bf16` and a prefill chunk of 0 are the slower, more reference-faithful opt-ins.

| Setting | Engine flag | Values | Default (blank) |
|---|---|---|---|
| `accel` | `--accel` | auto, cuda, cpu | auto (calibration picks the fastest correct lane) |
| `expand` | `--expand` | auto, never, always | auto (DTC, with EXPAND where predict asks) |
| `full_path` | `--full-path` | fast, exact | fast |
| `host_threads` | `--host-threads` | 1 to 8 | 1 (end to end the same within 1%: the lane is GPU-bound) |
| `tok_overlap` | `--tok-overlap` | on, off | on |
| `lm_head` | `--lm-head` | auto, int8, bf16 | auto (int8 on a 4 GB card) |
| `prefill_chunk` | `--prefill-chunk` | 0 to 512 | 64 (batched prefill) |
| `dtc_margin`, `dtc_entropy`, `dtc_prompt_tokens`, `dtc_complexity`, `dtc_longform` | `--dtc-*` | numbers, on/off | 0.5, 4.0, 96, 6, on |

**cuda_pack** (under paths) is the `.nq2` pack the CUDA engine runs from.

**Show console windows** (default off) lets the engine and helper programs open their own console
window. With it off they run with no window at all (`CREATE_NO_WINDOW`) and their output goes to the
**Engine log...** viewer in the dialog.

**Voice input (Parakeet)**: `parakeet_exe` and `parakeet_model` override where the mic button looks for
`parakeet-c.exe` and the model ([DICTATION.md](DICTATION.md#in-the-chat-window)).

**About and credits...** in the sidebar shows who made NTSE (the short form of
[CREDITS.md](CREDITS.md)).

## C scaler

Icons, buttons, banners, mascot strips and status dots are scaled by a small pure C
library, `littlenemo/gui/native/` (`nemo_scale.c`), loaded from Python with `ctypes`
(`gui/nemo_scale.py`):

- RGBA8 with premultiplied alpha; filters `NEMO_SCALE_NEAREST` (integer snap, used by
  Pixel Nemo), `NEMO_SCALE_LANCZOS3` and `NEMO_SCALE_MITCHELL`.
- Bounded and overflow-checked: `MAX_DIM` 8192, `MAX_PIXELS` 4096 x 4096.
- No globals, no locks in C. The Python wrapper keeps a scaled-image cache as a
  lock-free snapshot dictionary: readers hold the current dict, writers publish a
  new one, and a generation number keeps `clear_cache` from being overwritten by a
  late put. There is no lock in the tree ([ZERO-MUTEX.md](ZERO-MUTEX.md)).
- If the library is missing, the GUI falls back to the nearest prebuilt asset size and
  does not resample in Python.

Build and test (MSYS2 UCRT64 gcc on PATH):

```
cd littlenemo/gui/native
build.bat
test_nemo_scale.exe        # ALL PASS
```

On Linux: `./build.sh && ./test_nemo_scale`. The shared library lands next to the
sources (`nemo_scale.dll`, or `libnemo_scale.so`). It is a build product and is not
committed.

## DPI and resizing

The window is per-monitor DPI aware on Windows (Per-Monitor v2, falling back to
per-monitor and then system awareness). Fonts, pane widths, the minimum size and cached
art follow a DPI change when the window moves between monitors. The default size is
1560x940 scaled by DPI, capped to the screen, and the minimum size is a floor scaled by
DPI so the layout never collapses. Chat bubbles re-wrap on resize.

## Protocol v3 message reference

The wire format is [littlenemo/PROTOCOL.md](../littlenemo/PROTOCOL.md): one UTF-8 line per
message, text pieces escaped (`\\`, `\n`, `\t`), key=value values on v3 lines
percent-encoded, lines over 256 KiB ignored. v3 is additive: a v2 engine works unchanged,
and a v2 client ignores the new lines.

**Client to engine**

| Message | Meaning |
|---|---|
| `PROMPT <text>` | one user turn |
| `STOP` | end the current generation early |
| `RESET` | clear the conversation (and pending attachments) |
| `QUIT` | end the session |
| `MEDIA id= type= mime= bytes= [w= h= frames= dur_ms= sent_frames=] name= path=` | announce one attachment for the next PROMPT |
| `FRAME id= idx= ts_ms= path=` | one extracted video frame (PNG) for a MEDIA id |

**Engine to client**

| Message | Meaning |
|---|---|
| `READY nemo-engine <ver> model= weights= layers= cores= threads= mode= simd=` | engine is up; v3 adds `media=`, `accel=`, `moe=`, `fixture=1` |
| `THINK <text>` | a piece of the reasoning trace |
| `TOK <text>` | a piece of the visible answer |
| `LAYER <idx> <type>` | the forward pass entered a layer (`mamba`, `mlp`, `attention`) |
| `TICK mode= layer= next= active= staged= tok= ms_tok= gbs=` | per-token telemetry |
| `PREFILL tokens= ms= tps=` | prompt processing, once per turn |
| `STATS tokens= ms= tps= ms_per_tok= gbs_avg= peak_mb=` | end-of-turn summary; may add `prompt_tokens`, `prompt_ms`, `prompt_tps`, `ttft_ms`, `gen_tps`, `think_tokens`, `answer_tokens`, `stopped` |
| `DONE` | the turn is complete |
| `ERROR <text>` | the turn is over; the state is reset |
| `MEDIA_OK id= [type= tokens=]` | the attachment was accepted |
| `MEDIA_ERR id= reason=` | the attachment was refused |
| `HW key=value...` | hardware the engine runs on |
| `EXPAND layer= tensor= [rows= cols= bytes= lane=] state=begin\|end` | a tensor being expanded |
| `DTC layer= tensor= [rows= cols= k= block= active= lane=] state=begin\|end` | a tensor in dense tensor compute |
| `HS pass= ordinal= hits= misses= stall_ms= dtc= dtc_ms= [staged=]` | Tensor Handshake counters, cumulative per session |
| `MEM ram_mb= peak_mb= [vram_mb= vram_peak_mb=]` | memory |
| `MOE layer= experts= [k= of=]` | the experts routed for the current token |
| `STAT key=value...` | any other statistic, shown under "Other engine stats" |

## Modules

| File | Role |
|---|---|
| `nemo_chat.py` | entry point, INI and command line, the protocol client and engine process |
| `nemo_ui.py` | the Tk window, thinking line, dialogs |
| `nemo_stats.py` | stats model behind the panel |
| `nemo_media.py` | attachment inspection and limits |
| `nemo_themes.py`, `nemo_icons.py` | theme pack and icon pack loaders |
| `nemo_shapes.py` | the procedural shape renderer |
| `nemo_appearance.py` | theme, icon and shape settings in the INI (no Tk) |
| `nemo_scale.py` | ctypes front end to the C scaler |
| `nemo_dictation.py` | Parakeet dictation client |
| `nemo_proc.py` | spawn flags: hidden console windows by default, `engine_kwargs` and `hidden_kwargs` |
| `nemo_about.py` | the About and credits dialog |
| `nemo_chat.pyw` | windowless launcher for `nemo_chat.py` (`pythonw`) |
| `nemo_demo.py` | scripted demos for the smoke test and screenshots |

## Testing

Nine Python suites cover the GUI; they run from `littlenemo/tests` and use the fixture
engine ([TESTING.md](TESTING.md)). The icons suite needs the C scaler built first.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| window opens, every model shows unavailable | no engine configured: set `[paths]` in `littlenemo.ini`, or pass `--engine` |
| "SuperNemo engine not available yet" | expected: there is no SuperNemo engine yet |
| attachments refused | the selected model's READY has no `media` key (Little Nemo is text only), or a limit above was exceeded |
| video attachment fails | install `ffmpeg` and put it on PATH, or use an animated GIF or WebP |
| the engine seems to do nothing and no console shows | consoles are hidden by default: open **Settings... > Engine log...**, or tick **Show console windows** |
| "mic: parakeet-c not found (Settings, parakeet_exe)" | set `parakeet_exe` in Settings, or see [DICTATION.md](DICTATION.md#troubleshooting) |
| icons look blurry or are missing sizes | the C scaler is not built; run `build.bat` in `gui/native` |
| `--chat` is not applied | the INI-built command cannot carry it; use `--engine` with the full command |
| engine exits before READY | the checkpoint failed validation; the engine names the tensor on stderr |
| dictation does nothing | see [DICTATION.md](DICTATION.md) troubleshooting |
| a theme does not appear | check the folder is `themes/<name>/theme.json` beside the INI and that the JSON is valid |