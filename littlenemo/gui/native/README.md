# NTSE GUI native scaler

A small pure-C RGBA8 scaler used by the Tk GUI for icons, buttons,
banners, thinking strips / mascots and status dots.

- Premultiplied alpha.
- Filters: `NEMO_SCALE_NEAREST` (integer-snap, Pixel Nemo),
  `NEMO_SCALE_LANCZOS3`, `NEMO_SCALE_MITCHELL`.
- Safe-critical: bounded dimensions, overflow-checked sizes, no
  globals, no locks.
- Python talks to the library through ctypes (`gui/nemo_scale.py`).
  If the DLL / `.so` is missing, the GUI falls back to the nearest
  prebuilt asset size and does **not** resample in Python.

## Build

Use the compiler already on the machine. Do not install anything.

    # Linux / the box
    ./build.sh && ./test_nemo_scale

    # TheWorkstation (MSYS2 UCRT64 gcc on PATH, or:)
    MSYS2 UCRT64 gcc on PATH  (see build.bat)
    build.bat && test_nemo_scale.exe

The shared library lands next to the sources: `nemo_scale.dll` on
Windows, `libnemo_scale.so` elsewhere.
