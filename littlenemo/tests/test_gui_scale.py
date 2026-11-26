#!/usr/bin/env python3
"""test_gui_scale.py - Python front end of the native RGBA8 scaler
(ctypes, cache, source-size picker, missing-DLL fallback). The C
unit tests live in gui/native/test_nemo_scale.c and are run separately.
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gui"))

import nemo_scale as sc  # noqa: E402
from PIL import Image  # noqa: E402

fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def main():
    expect(sc.pick_source_size([16, 24, 32, 48, 64], 40) == 48,
           "picker takes the smallest prebuilt size >= dest")
    expect(sc.pick_source_size([16, 32], 32) == 32,
           "picker prefers an exact match")
    expect(sc.pick_source_size([16, 32], 128) == 32,
           "picker falls back to the largest when dest is bigger")
    expect(sc.pick_source_size([], 32) is None, "picker of an empty set is None")
    expect(sc.filter_for_theme({"id": "pixel", "scale": "nearest"}) == sc.NEAREST
           and sc.filter_for_theme({"id": "lattice"}) == sc.LANCZOS3,
           "Pixel Nemo uses nearest; smooth themes use Lanczos3")

    # Premul / unpremul round trip for opaque and transparent.
    raw = bytes([10, 20, 30, 255, 0, 0, 0, 0, 64, 0, 0, 128])
    pre = sc.premultiply(raw, 3, 1)
    expect(pre[0] == 10 and pre[3] == 255 and pre[4:8] == b"\0\0\0\0"
           and pre[8] == (64 * 128 + 127) // 255 and pre[11] == 128,
           "premultiply leaves opaque/zero alone and scales partial alpha")
    back = sc.unpremultiply(pre, 3, 1)
    expect(back[0] == 10 and back[3] == 255 and back[4:8] == b"\0\0\0\0"
           and back[11] == 128 and abs(back[8] - 64) <= 1,
           "unpremultiply recovers straight alpha (within 1)")

    if not sc.available():
        print("note: native scaler not loaded; skipping live scale checks")
        expect(sc.scale_rgba(b"\0" * 16, 2, 2, 4, 4, "nearest") is None,
               "scale_rgba returns None when the DLL is missing")
        expect(sc.scale_image(Image.new("RGBA", (2, 2), (1, 2, 3, 4)),
                              4, 4, "nearest") is None,
               "scale_image returns None when the DLL is missing")
        print("RESULT:", "FAILURES" if fails else "ALL PASS")
        return 1 if fails else 0

    expect(sc.version() and sc.version().startswith("ntse-scale"),
           "native version is readable")
    expect(os.path.isfile(sc.library_path() or ""), "library path exists")

    # Identity through the wrapper.
    im = Image.new("RGBA", (5, 5))
    for y in range(5):
        for x in range(5):
            im.putpixel((x, y), (x * 40, y * 40, 80, 200))
    out = sc.scale_image(im, 5, 5, "lanczos3")
    expect(out is not None and out.size == (5, 5), "identity scale returns 5x5")
    # Opaque-ish pixels should survive a premul round trip closely.
    px = out.getpixel((2, 3))
    src = im.getpixel((2, 3))
    expect(all(abs(a - b) <= 1 for a, b in zip(px, src)),
           "identity scale is a near-exact round trip")

    # Nearest 2x exactness on a 2x2 of opaque colours.
    tiny = Image.new("RGBA", (2, 2))
    tiny.putpixel((0, 0), (255, 0, 0, 255))
    tiny.putpixel((1, 0), (0, 255, 0, 255))
    tiny.putpixel((0, 1), (0, 0, 255, 255))
    tiny.putpixel((1, 1), (0, 0, 0, 0))
    big = sc.scale_image(tiny, 4, 4, "nearest")
    expect(big is not None and big.size == (4, 4), "nearest 2x returns 4x4")
    expect(big.getpixel((0, 0)) == (255, 0, 0, 255)
           and big.getpixel((1, 0)) == (255, 0, 0, 255)
           and big.getpixel((2, 0)) == (0, 255, 0, 255)
           and big.getpixel((3, 3)) == (0, 0, 0, 0),
           "nearest 2x replicates each source pixel into a 2x2")

    # Cache + load_sized: writes two prebuilt sizes, asks for a third.
    tmp = tempfile.mkdtemp(prefix="nemo-scale-")
    p16 = os.path.join(tmp, "i-16.png")
    p64 = os.path.join(tmp, "i-64.png")
    Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(p16)
    Image.new("RGBA", (64, 64), (40, 50, 60, 255)).save(p64)
    table = {16: p16, 64: p64}
    got = sc.load_sized(table, 32, "nearest")
    expect(got is not None and got.size == (32, 32),
           "load_sized scales the next-larger (64) down to 32")
    # Cache hit path (same file, same dest).
    sc.clear_cache()
    a = sc.load_sized(table, 48, "lanczos3")
    b = sc.load_sized(table, 48, "lanczos3")
    expect(a is not None and b is not None and a.size == b.size == (48, 48),
           "cached load_sized returns the requested size twice")

    # Fallback contract: when available() is forced off, load_sized
    # must not resample (it returns the 64 px prebuilt for dest 48).
    sc._lib_tried = True
    saved, sc._lib = sc._lib, None
    try:
        got = sc.load_sized(table, 48, "lanczos3")
        expect(got is not None and got.size == (64, 64),
               "missing DLL: load_sized returns the nearest prebuilt, unscaled")
        expect(sc.scale_image(tiny, 8, 8, "nearest") is None,
               "missing DLL: scale_image returns None (no Python resample)")
    finally:
        sc._lib = saved
        sc._lib_tried = True

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
