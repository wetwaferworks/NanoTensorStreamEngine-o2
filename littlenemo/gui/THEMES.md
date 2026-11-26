# NTSE chat themes: format

The official theme pack lives at `gui/theme-pack/` and is loaded
**directly** from `theme.json` (the CloudAtlas v1 format). JD has
approved all five themes; all five are selectable. The default is
**Pixel Nemo** (`default_theme: pixel`).

| id          | Name            | Subtitle        | Folder                     | Default |
| ----------- | --------------- | --------------- | -------------------------- | ------- |
| `pixel`     | Pixel Nemo      | Retro Pixel     | `themes/5-pixel-nemo/`     | **yes** |
| `lattice`   | Lattice         | Tensor Crystal  | `themes/1-lattice/`        |         |
| `stream`    | Stream          | Bitstream       | `themes/2-stream/`         |         |
| `handshake` | o.O Handshake   |                 | `themes/3-handshake-oO/`   |         |
| `owl`       | Sage Owl        |                 | `themes/4-sage-owl/`       |         |

The pack's `scripts/` folder is **build tooling only** and is not
shipped with the GUI. Do not copy it into the worktree.

The GUI applies a theme **live**, without restarting the engine or the
window. Invalid values are skipped so a typo cannot crash the UI. The
window icon, tray icon, composer / sidebar glyphs, status dots and
theme-picker banners all come from the active theme's assets.

If the pack is missing, the picker falls back to two built-in themes
(Night and Ember, also kept as `gui/themes/*.json` for reference).
Those are hidden once the official pack is present.

## The Handshake banner tag

The Handshake banner (`themes/3-handshake-oO/banner/banner-1200x300.png`)
carries a small product-name tag in the bottom-right corner. That name
has not been cleared as a public mark, so these docs do not reproduce
it. The other Handshake assets (window icon, buttons, thinking strip,
status dots, tile) do not carry the tag.

## Pack layout

```
gui/theme-pack/
  theme.json
  README.md
  sheets/                  # review sheets (optional at runtime)
  themes/<n-name>/
    icon/ntse.ico          # 16, 24, 32, 48, 64, 256
    icon/icon-{16,24,32,48,64,128,256,512}.png
    tray/tray-{16,32}.png
    banner/banner-1200x300.png
    buttons/plated/<action>-{24,48}.png
    buttons/glyph/<action>-{24,48}.png
    thinking/thinking-strip-16x16x12.png
    status/{ready,thinking,generating,expanding,dense-compute,error}-{16,32}.png
    tile/bg-tile-*.png
```

The ten button actions are: `send`, `stop`, `attach-image`,
`attach-video`, `attach-file`, `new-chat`, `settings`, `model-switch`,
`copy`, `clear`. Glyph-only buttons are bare symbols on transparency,
made for dark surfaces; plated buttons include the theme's button
shape. The send glyph uses the orange accent in every theme.

The thinking strip is 12 frames of 16×16 in a horizontal PNG, at about
80 ms per frame, looping. Pixel Nemo is upscaled with nearest-neighbour
so the pixels stay square; the other themes use a smooth scale.

Status colours are shared across all five themes: ready `#4ADE80`,
thinking `#A970FF`, generating `#FF9A3C`, expanding `#38BDF8`,
dense compute `#FFE066`, error `#FF5C6C`. Each status also has its
own shape mark, so colour is not the only cue. GUI animation states
map as: idle/done → ready, thinking → thinking, streaming → generating,
expanding → expanding, dtc → dense compute, error → error.

## `theme.json`

The file at the pack root names the app, pack version, date,
`default_theme`, shared `status_colors`, and a `themes` array. Each
entry has:

- `id`, `order`, `name`, `subtitle`, `description`, `default`, `folder`
- `colors`: `bg`, `panel`, `panel2`, `line`, `text`, `muted`, `p`, `p2`, `o`
- `fonts`: `ui` and `heading` aliases (`vt`, `press`, `chakra`, `inter`,
  `space`, `jb`, `exo2`). The GUI maps these to installed families and
  **never downloads a font**. If a family is not installed, the next
  alias in the list is tried, then a safe system font (e.g. Segoe UI /
  Consolas).
- `corner_radius` and `status_dot_shape`
- `assets`: relative paths for `ico`, `icon_png`, `tray`, `banner`,
  `buttons`, `thinking_strip`, `status`, `background_tile`

The pack colour keys are mapped onto the GUI palette (`bg` → window,
`panel` → sidebar and stats, `panel2` → composer / bubbles, `line` →
borders, `text` / `muted`, `p` → accent purple, `p2` → thinking text,
`o` → accent orange). Extra GUI roles (`raised`, `select`, `warn_bg`,
and so on) are derived from those.

All five official themes share `p` `#A970FF`, `p2` `#D9BFFF`,
`o` `#FF9A3C`. Surfaces and typography are what distinguish them:
Pixel Nemo is radius 0 with VT / Press Start; Lattice is radius 4
hexagon with Chakra; Stream is radius 16 with Inter / Space Grotesk;
Handshake is radius 10 circle with JetBrains Mono; Sage Owl is
radius 12 diamond with Exo 2.

## Older single-theme JSON (still accepted)

A drop-in `gui/themes/<id>.json`, or a `themes/` folder beside the
user's `littlenemo.ini`, can still add a theme in the older format
(a single object with `palette` / `fonts` / `spacing`). A drop-in
cannot override an official pack id. Night and Ember in `gui/themes/`
are the fallbacks used only when the pack is absent; they are
documented here so a typo in a custom file still has a complete
palette to inherit from.

```json
{
  "id": "reef",
  "name": "Reef",
  "author": "CloudAtlas",
  "version": 1,
  "palette": {
    "bg": "#0E0E10",
    "sidebar": "#151517",
    "panel": "#141416",
    "surface": "#1D1D21",
    "raised": "#26262B",
    "border": "#2E2E34",
    "select": "#2A2438",
    "fg": "#ECECF1",
    "fg2": "#C9C9D1",
    "dim": "#8E8E98",
    "faint": "#5F5F69",
    "accent": "#A970FF",
    "accent2": "#FF9B42",
    "think": "#C4A6FF",
    "think_hot": "#E8DEFF",
    "ok": "#4ADE80",
    "warn": "#F5B041",
    "warn_bg": "#2B2210",
    "error": "#FF7A7A",
    "user_bubble": "#2A2438",
    "assistant_fg": "#ECECF1",
    "icon_fill": "#A970FF",
    "icon_outline": "#FF9B42"
  },
  "fonts": {
    "family": "Segoe UI",
    "mono": "Cascadia Mono",
    "body": 11,
    "small": 9,
    "title": 13,
    "hero": 20
  },
  "spacing": {
    "sidebar_width": 236,
    "stats_width": 372,
    "pad": 16,
    "composer_pad": 40,
    "icon": 32,
    "icon_lg": 64,
    "radius": 8
  },
  "icons": "default"
}
```

Every palette value is `#RGB` or `#RRGGBB`. Anything else is ignored.
Font sizes are integers from 6 to 48 (points). Spacing values are
integers from 4 to 800 (logical pixels, scaled for DPI). `icons` may
be a pack id or a path (see ICONS.md); a theme's thinking strip is
also offered in the icon picker as `theme-strip:<id>`.

## Hooking a theme up

The active theme is `theme = <id>` under `[appearance]` in
`littlenemo.ini`, or the choice in Appearance → Themes. Switching in
the picker applies immediately (window icon included) and writes that
key. An unknown or missing id falls back to Pixel Nemo when the pack
is present, or to Night if it is not.

## Scaling

Art is scaled in C (`gui/native/nemo_scale.c`, called through ctypes),
never in Python. Smooth themes use Lanczos3 (Mitchell is also
available); Pixel Nemo uses nearest-integer mode so 2x / 3x / 4x stay
crisp. The GUI picks the nearest larger prebuilt size and caches the
result per DPI and dest size. If the DLL is missing, it uses the
nearest prebuilt size as-is and does not resample.
