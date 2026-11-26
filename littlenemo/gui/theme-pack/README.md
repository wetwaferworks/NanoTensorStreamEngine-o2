# NTSE GUI themes v1 (full asset set, 2026-11-26)

Five themes for the NTSE (Nano Tensor Streaming Engine) desktop GUI, which is a Tkinter app on Windows with a sidebar, a chat panel and a live stats panel. JD has confirmed all five themes. All five can be selected in the app.

**Default theme: 5 · Pixel Nemo.** In `theme.json`, `"default_theme": "pixel"` is set, and Pixel Nemo has `"default": true`.

| # | Theme | Folder | Default |
|---|-------|--------|---------|
| 5 | **Pixel Nemo: Retro Pixel** | `themes/5-pixel-nemo/` | **yes** |
| 1 | Lattice: Tensor Crystal | `themes/1-lattice/` | |
| 2 | Stream: Bitstream | `themes/2-stream/` | |
| 3 | o.O Handshake | `themes/3-handshake-oO/` | |
| 4 | Sage Owl | `themes/4-sage-owl/` | |

## What each theme folder contains
- `icon/ntse.ico`: holds the 16, 24, 32, 48, 64 and 256 px images. Each size is drawn on its own, not scaled down from one image, and each was checked pixel for pixel against its PNG.
- `icon/icon-{16,24,32,48,64,128,256,512}.png`
- `tray/tray-{16,32}.png`: the glyph without its tile, with a 1 px dark halo so it reads on both light and dark taskbars.
- `banner/banner-1200x300.png`
- `buttons/plated/<name>-{24,48}.png` and `buttons/glyph/<name>-{24,48}.png`. The ten names are send, stop, attach-image, attach-video, attach-file, new-chat, settings, model-switch, copy and clear.
  - Plated buttons include the theme's button shape.
  - Glyph-only buttons are the bare symbol on transparency, made for the dark UI.
  - The send button uses the orange accent in every theme.
- `thinking/thinking-strip-16x16x12.png`: 12 frames of 16x16 in a horizontal strip. About 80 ms per frame is suggested, looping.
- `status/{ready,thinking,generating,expanding,dense-compute,error}-{16,32}.png`. The six colors are the same in every theme. Each status also has its own shape mark, so color is not the only cue.
- `tile/bg-tile-*.png`: a seamless, opaque background tile in the theme's base color.

`theme.json` lists each theme's colors, fonts, corner radius, dot shape and default flag, plus the path to every asset.

## Review sheets (3840x2160)
- `sheets/sheet-N-*.png`: one sheet per theme.
- `sheets/overview-all-5.png`: all five themes side by side, with Pixel Nemo first.

## Notes
- All text colors pass WCAG AA on each theme's base and panel colors. Text is 15:1 or better, and muted text is 6.7:1 or better.
- There are no Grok or xAI logos or branding.
- Pixel Nemo stays on a whole-pixel grid at 16, 32, 48, 64, 128, 256 and 512 (the 16x16 sprite scaled by 1x to 32x). Its 24 px icon puts the fish at 1x on a 24 px tile so the pixels stay square. Pixel buttons are 12x12 glyphs, scaled 2x at 24 px and 4x at 48 px for glyph-only. The plated versions use a frame drawn at 1x (24 px) or 3x (48 px).
- The o.O Handshake 16 px icon and 16 px tray icon are hand-placed pixel by pixel, with o, dot and O sitting on a shared baseline.
- To rebuild everything, run `python3 scripts/ntse_build_v1.py` (needs PIL only). The 2026-11-26 preview set stays unchanged in `../ntse-gui-art-2026-11-26/`.

Credits: art by CloudAtlas · Technical Art Director. The o.O mark is by JD.
