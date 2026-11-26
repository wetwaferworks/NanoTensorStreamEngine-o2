# NTSE mascots v1 (2026-11-26)

These are animated model mascots for the NTSE thinking line. There is one pack per theme, and **Pixel Nemo is the default**. All five packs can be selected.

| Pack | Folder | Theme | Default |
|------|--------|-------|---------|
| Pixel Nemo (clownfish) | `pixel-nemo/` | pixel | **yes** |
| Lattice Cube | `lattice-cube/` | lattice | |
| Stream Bit Critter | `stream-critter/` | stream | |
| o.O Handshake (two rings) | `handshake-oO/` | handshake | |
| Sage Owl | `sage-owl/` | owl | |

Each pack folder contains:
- `manifest.json`
- `<state>-32.png` and `<state>-64.png` for each of the seven states
- `preview-<pack>.gif`: a 4x preview that cycles through every state

All art is pixel art drawn at 32 px. The 64 px files are exact 2x nearest-neighbour copies. Backgrounds are transparent RGBA, and every sprite has a 1 px dark outline so it reads on both dark and light backgrounds.

## States (the same motion design in every pack)
| state | frames | fps | motion | what it does |
|-------|--------|-----|--------|--------------|
| idle | 8 | 6 | bob | slow float with one blink |
| thinking | 8 | 10 | wobble | side-to-side tilt, eyes look up, thought dots rise |
| streaming | 8 | 14 | bob | quick forward bob while bit packets stream out behind |
| expanding | 8 | 10 | pulse | swells outward with a growing ring |
| dtc | 12 | 16 | spin | full spin on its vertical axis while tensor blocks fire in the corners (unlike expanding, nothing grows) |
| error | 8 | 12 | wobble | shake, red flash, X eyes and a "!" |
| done | 10 | 12 | flip | happy hop with a flip at the top and a sparkle landing; plays once and holds the last frame |

## Sheets
- `sheets/sheet-<pack>.png` (3840x2160): every state's frames at 32 px actual size, the 64 px strip, each frame zoomed 4x, and the 64 px frames on a light background.
- `sheets/overview-all-5.png`: all five packs side by side.

## Manifest schema `ntse-mascot/1` (for third-party model packs)
```json
{
  "schema": "ntse-mascot/1",
  "id": "my-model",                 // unique, lowercase, a-z 0-9 and -
  "name": "My Model",
  "theme": "pixel",                 // optional: preferred NTSE theme id
  "default": false,                 // only the built-in Pixel Nemo pack sets true
  "author": "Your name",
  "sizes": [32, 64],                // 32 is required; 64 is optional (otherwise the 32 px art is scaled 2x)
  "frame_size": {"32": [32, 32], "64": [64, 64]},
  "layout": "horizontal-strip",     // or "frames" with numbered files: <state>-<size>-00.png ...
  "scaling": "nearest-integer",
  "color": "RGBA, transparent background",
  "fallback_state": "idle",         // used for any state the pack leaves out
  "states": {
    "idle": {
      "frames": 8,                  // 1-12 (6-12 recommended)
      "fps": 6,                     // optional, 1-20 (20 is the maximum)
      "motion": "bob",              // optional: flip | bob | wobble | spin | pulse
      "loop": true,                 // optional, default true; false = play once, hold last frame
      "strip": {"32": "idle-32.png", "64": "idle-64.png"},
      "description": "optional text"
    }
  }
}
```
Rules:
- **State names:** idle, thinking, streaming, expanding, dtc, error, done. Only `idle` is required, and any missing state falls back to `fallback_state`.
- **Strips:** each strip is exactly `frames × size` wide and `size` tall, left to right, with no padding.
- **Files:** paths are relative to the pack folder. Use PNG files only, no scripts.

Comments in the JSON above are for explanation only, so remove them in a real manifest.

To rebuild everything, run `python3 scripts/mascots.py` (needs PIL only).

Credits: mascots by CloudAtlas · Technical Art Director. The HandshakeOS name and o.O mark are by JD.
