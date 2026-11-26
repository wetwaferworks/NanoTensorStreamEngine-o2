# NTSE model icons: asset format

Placeholder packs live in `gui/icons/supernemo/` and `gui/icons/littlenemo/`.
They are procedurally drawn stand-ins (simple shapes, JD's purple `#A970FF`
and orange `#FF9B42` on a transparent background). Replace the PNGs in place
to drop in the real art; if you keep the same frame counts, you can leave
`manifest.json` as it is.

The GUI never requires these files to start: the same shapes are also
rendered in memory as a built-in fallback. A missing, unreadable or
malformed pack falls back to the model's procedural thinking shape, then
to the built-in default. It will not crash the window.

Official theme thinking strips (16×16×12, from the CloudAtlas theme pack)
are also pickable as `theme-strip:<id>` packs. They do not silently
replace a model's mascot; the user (or a saved `icon=`) has to choose
them. See THEMES.md.

## Where packs are found

1. `icon=` on a `[model.<key>]` section of `littlenemo.ini`, as an
   absolute path, a path relative to the INI, or a path relative to
   `gui/icons/`. `builtin:<key>` and `theme-strip:<theme-id>` are also
   accepted.
2. Any pack folder (or `.png` strip) inside `gui/icons/` or a `icons/`
   folder beside the settings file — these appear in the icon picker,
   along with each official theme's thinking strip.
3. Built-in ids: `builtin:supernemo`, `builtin:littlenemo`, `builtin:default`.

An empty `icon=` means "use the procedural thinking shape" (see the
shape customizer). A valid PNG pack **overrides** the procedural shape;
the shape settings are kept so clearing the pack restores them.

## Sizes

Every pack should ship **32 px** and **64 px** frames (square, RGBA,
transparent background). Those are the only sizes the mascot player
uses.

- 32 px is used on the thinking line and in the sidebar.
- 64 px is used in the picker, the shape-customizer preview, and on
  high-DPI displays (the 32 px art is scaled up if 64 is missing, but
  painted 64s look better).

Colour art should sit in the centre with a few pixels of padding so a
bob or pulse does not clip. A good safe area is a 28×28 (or 56×56)
drawing inside the frame.

Theme thinking strips start at 16×16 and are scaled up to 32 and 64
(nearest-neighbour for Pixel Nemo, smooth for the others). A painted
32/64 mascot pack is still the right way to ship a personality.

## Two layouts (both are accepted)

### Frame strip

A single horizontal PNG: `N` square frames side by side. Vertical
strips (taller than they are wide) are also accepted. Files:

```
my-pack/
  manifest.json
  strip-32.png      # width = 32 × N, height = 32
  strip-64.png      # width = 64 × N, height = 64
```

### Numbered frames

```
my-pack/
  manifest.json
  frames-32/000.png  001.png  002.png  ...
  frames-64/000.png  001.png  002.png  ...
```

Names are sorted lexicographically; zero-padded three-digit names
(`000.png`) keep the order obvious. You may ship both a strip and a
frames folder; the manifest decides which to prefer (`"format": "strip"`
or `"format": "frames"`). The placeholder packs ship both, so you can
replace either.

A bare `strip.png` or a lone `.png` path in `icon=` is also valid: the
GUI splits it into square frames and spreads them across states.

## `manifest.json`

```json
{
  "name": "SuperNemo",
  "author": "CloudAtlas",
  "note": "optional, shown nowhere required",
  "sizes": [32, 64],
  "fps": 8,
  "format": "strip",
  "layout": "horizontal",
  "strip":  { "32": "strip-32.png", "64": "strip-64.png" },
  "frames": { "32": "frames-32",    "64": "frames-64" },
  "states": {
    "idle":      { "frames": [0],           "fps": 1,  "loop": true },
    "thinking":  { "frames": [1, 2, 3, 4],  "fps": 8,  "loop": true, "motion": "bob" },
    "streaming": { "frames": [3, 4, 5],     "fps": 10, "loop": true, "motion": "wobble" },
    "expanding": { "frames": [4, 5, 1, 2],  "fps": 8,  "loop": true, "motion": "spin" },
    "dtc":       { "frames": [5, 4, 2, 1],  "fps": 8,  "loop": true, "motion": "pulse" },
    "error":     { "frames": [6],           "fps": 1,  "loop": false },
    "done":      { "frames": [7],           "fps": 1,  "loop": false }
  }
}
```

Rules:

- `states` keys are the **seven** names above. Unknown keys are ignored.
  A missing state is filled from a sensible default (idle / error / done
  hold the first frame; motion states cycle the rest).
- `frames` is a list of **0-based indices** into the strip / numbered
  files. Indices must land inside the pack (an out-of-range index
  rejects the whole pack and the GUI falls back).
- `fps` is a number in `(0, 60]`. The GUI caps playback at 20 fps so
  the thinking line stays cheap on CPU.
- `loop` defaults to `true`. `error` and `done` should usually be
  `false` (hold the last frame).
- `motion` is optional and one of `flip`, `bob`, `wobble`, `spin`,
  `pulse`. It is a small transform applied on top of the frames, so a
  one-frame pack can still nod and flip. If you paint the motion into
  the frames yourself, omit `motion` (or the painted motion and the
  transform will stack).
- `name` is what the picker shows. `id` is the folder path (or the
  `builtin:…` / `theme-strip:…` id).

If `manifest.json` is missing, the first frame is used for idle / error
/ done and the rest cycle for thinking, streaming, expanding and dtc.

## Animation states (driven by the engine protocol)

There are **seven** states. `expanding` and `dtc` are distinct: EXPAND
and DTC are different protocol events, and the motions stay visually
different even if the user already picked spin or pulse for thinking.

| State       | When                                                         | Suggested feel                          |
| ----------- | ------------------------------------------------------------ | --------------------------------------- |
| `idle`      | No turn in flight                                            | Rest pose, eyes forward. No motion.     |
| `thinking`  | `THINK` lines, or after Send before the first `TOK`          | Looking up, a bob / flip / wobble.      |
| `streaming` | `TOK` lines (the answer is landing)                          | A quicker, lighter motion (wobble).     |
| `expanding` | `EXPAND` with `state=begin` (tensor work in flight)          | A **distinct** motion (spin, or pulse if the user already chose spin). |
| `dtc`       | `DTC` with `state=begin` (dense-compute work in flight)      | A **different** distinct motion (pulse, or spin if the user already chose pulse). |
| `error`     | `ERROR` (or a failed turn)                                   | Frown / X eyes; short wobble, then hold.|
| `done`      | `DONE`                                                       | A small happy pulse, then rest.         |

`EXPAND`/`DTC` with `state=end` returns to thinking or streaming (or
to the other of expanding/dtc if that is still in flight). The
timer stops when the window is hidden, and after `done`/`error` settle,
so an idle chat does not spend CPU. Animation is Tk `after()` only; no
threads, no locks.

## Style notes for the real art

- These are *personalities*, little shapes that look like they are
  thinking, not logos. Do not copy Grok / xAI marks or any other
  brand.
- Default palette: purple `#A970FF`, orange `#FF9B42`, dark ink
  `#1A1228`, eye-white `#F7F4FF`, on a **transparent** background so
  the icon sits on any theme.
- A face (eyes, a tiny mouth) reads clearly at 32 px. SuperNemo's
  placeholder is a purple blob; Little Nemo's is an orange droplet.
  New models can be any of blob, orb, cube, star, ring, droplet, or
  a fully painted pack.
- Keep the silhouettes distinct from each other in the sidebar at 32 px.

## Hooking a pack up

In `littlenemo.ini`:

```
[model.supernemo]
icon = supernemo
; or an absolute path, or builtin:supernemo, or theme-strip:pixel
```

Or pick it in the Appearance dialog (sidebar → Appearance…, or the
small icon on a model row). The choice is saved in the same INI.

## Official mascots (`ntse-mascot/1`)

CloudAtlas's official mascots live in `gui/icons/` as one folder per
theme: `pixel-nemo` (the default), `lattice-cube`, `stream-critter`,
`handshake-oO` and `sage-owl`. Each pack uses the `ntse-mascot/1`
schema: a `manifest.json` plus `<state>-32.png` / `<state>-64.png`
horizontal strips for the seven states. `frames` in that schema is a
**count** (not a list of indices); each state has its own strip. Extra
keys (`loop`, `fallback_state`, `description`, `theme`, `scaling`,
`schema`) are accepted and ignored when the loader does not need them.
The same `motion` tag may appear on more than one state. A state the
pack leaves out uses `fallback_state` (default `idle`).

The 64 px art is a nearest-integer 2x of the 32 px art. When a model
has no `icon=` of its own, the thinking line (and the sidebar) uses
the mascot that matches the active theme.

The older single-strip / numbered-frames layout above is still fully
supported. Preview GIFs and pack `scripts/` are build tooling and are
not shipped in the worktree.

