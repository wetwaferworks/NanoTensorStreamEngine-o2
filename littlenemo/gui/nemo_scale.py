#!/usr/bin/env python3
"""nemo_scale.py - ctypes front end to the pure-C RGBA8 scaler.

The C library (gui/native/nemo_scale.dll or libnemo_scale.so) does
every resample: Lanczos3 / Mitchell for smooth art, nearest-integer
for Pixel Nemo. This module never resamples in Python. If the library
is missing, scale() returns None and the caller must use the nearest
prebuilt asset size as-is.

Input images are straight-alpha (as PNG / Pillow deliver them). They
are premultiplied for the C call and un-premultiplied on the way out
so Tk / Pillow can display them.
"""

import ctypes
import os

NEAREST = 0
LANCZOS3 = 1
MITCHELL = 2

FILTER_NAMES = {
    "nearest": NEAREST,
    "lanczos3": LANCZOS3,
    "lanczos": LANCZOS3,
    "mitchell": MITCHELL,
    "smooth": LANCZOS3,
}

# A tiny, process-local cache of scaled RGBA bytes. The C library has
# no globals; this cache lives only in Python and is keyed by
# (source identity, dest size, filter). The dict is replaced as a
# whole, so readers and writers never take a lock: a reader holds a
# local reference to the current snapshot, and a writer builds a new
# dict and binds it. A generation number keeps clear_cache from being
# overwritten by a late put. A lost concurrent put is fine for a
# cache (the next call just scales again). Callers on the Tk thread
# typically do not contend.
_cache = {}
_cache_gen = 0
_CACHE_MAX = 256

_lib = None
_lib_tried = False
_lib_path = None


def _candidates():
    here = os.path.dirname(os.path.abspath(__file__))
    native = os.path.join(here, "native")
    names = (["nemo_scale.dll"] if os.name == "nt"
             else ["libnemo_scale.so", "nemo_scale.so"])
    out = []
    for folder in (native, here, os.getcwd()):
        for name in names:
            out.append(os.path.join(folder, name))
    return out


def _load():
    global _lib, _lib_tried, _lib_path
    if _lib_tried:
        return _lib
    _lib_tried = True
    last = None
    for path in _candidates():
        if not os.path.isfile(path):
            continue
        try:
            lib = ctypes.CDLL(os.path.abspath(path))
            lib.nemo_scale_rgba8.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int,
            ]
            lib.nemo_scale_rgba8.restype = ctypes.c_int
            lib.nemo_scale_version.argtypes = []
            lib.nemo_scale_version.restype = ctypes.c_char_p
            # Probe it.
            lib.nemo_scale_version()
            _lib = lib
            _lib_path = os.path.abspath(path)
            return _lib
        except OSError as exc:
            last = exc
            continue
    _lib = None
    _lib_path = None
    return None


def available():
    """True if the native scaler loaded."""
    return _load() is not None


def library_path():
    _load()
    return _lib_path


def version():
    lib = _load()
    if lib is None:
        return None
    v = lib.nemo_scale_version()
    return v.decode("ascii", "replace") if v else None


def _norm_filter(filter):
    if isinstance(filter, int):
        if filter in (NEAREST, LANCZOS3, MITCHELL):
            return filter
        raise ValueError(f"unknown filter {filter!r}")
    name = (filter or "lanczos3").strip().lower()
    if name not in FILTER_NAMES:
        raise ValueError(f"unknown filter {filter!r}")
    return FILTER_NAMES[name]


def filter_for_theme(theme=None):
    """Nearest for Pixel Nemo (and any theme with scale=nearest);
    Lanczos3 otherwise."""
    if not theme:
        return LANCZOS3
    if theme.get("scale") == "nearest" or theme.get("id") == "pixel":
        return NEAREST
    return LANCZOS3


def pick_source_size(sizes, dest):
    """The smallest prebuilt size that is >= dest, else the largest
    available. sizes is an iterable of ints. Returns None if empty."""
    nums = sorted({int(s) for s in (sizes or []) if int(s) > 0})
    if not nums:
        return None
    dest = int(dest)
    for s in nums:
        if s >= dest:
            return s
    return nums[-1]


def premultiply(buf, width, height):
    """Straight RGBA8 bytes -> premultiplied RGBA8 bytes."""
    src = memoryview(buf)
    n = width * height * 4
    out = bytearray(n)
    for i in range(0, n, 4):
        r, g, b, a = src[i], src[i + 1], src[i + 2], src[i + 3]
        if a == 255:
            out[i] = r
            out[i + 1] = g
            out[i + 2] = b
            out[i + 3] = 255
        elif a == 0:
            out[i] = out[i + 1] = out[i + 2] = out[i + 3] = 0
        else:
            out[i] = (r * a + 127) // 255
            out[i + 1] = (g * a + 127) // 255
            out[i + 2] = (b * a + 127) // 255
            out[i + 3] = a
    return bytes(out)


def unpremultiply(buf, width, height):
    """Premultiplied RGBA8 bytes -> straight RGBA8 bytes."""
    src = memoryview(buf)
    n = width * height * 4
    out = bytearray(n)
    for i in range(0, n, 4):
        r, g, b, a = src[i], src[i + 1], src[i + 2], src[i + 3]
        if a == 255:
            out[i] = r
            out[i + 1] = g
            out[i + 2] = b
            out[i + 3] = 255
        elif a == 0:
            out[i] = out[i + 1] = out[i + 2] = out[i + 3] = 0
        else:
            out[i] = min(255, (r * 255 + a // 2) // a)
            out[i + 1] = min(255, (g * 255 + a // 2) // a)
            out[i + 2] = min(255, (b * 255 + a // 2) // a)
            out[i + 3] = a
    return bytes(out)


def scale_rgba(src, src_w, src_h, dst_w, dst_h, filter="lanczos3"):
    """Premultiplied RGBA8 `src` (bytes) -> premultiplied RGBA8 bytes,
    or None if the native library is missing or the call fails."""
    lib = _load()
    if lib is None:
        return None
    filt = _norm_filter(filter)
    src_w, src_h = int(src_w), int(src_h)
    dst_w, dst_h = int(dst_w), int(dst_h)
    need_src = src_w * src_h * 4
    if src_w <= 0 or src_h <= 0 or dst_w <= 0 or dst_h <= 0:
        return None
    if len(src) < need_src:
        return None
    src_buf = (ctypes.c_uint8 * need_src).from_buffer_copy(src[:need_src])
    dst_n = dst_w * dst_h * 4
    dst_buf = (ctypes.c_uint8 * dst_n)()
    rc = lib.nemo_scale_rgba8(src_buf, src_w, src_h, 0,
                              dst_buf, dst_w, dst_h, 0, filt)
    if rc != 0:
        return None
    return bytes(dst_buf)


def scale_image(im, dst_w, dst_h, filter="lanczos3"):
    """A straight-alpha Pillow image -> a new straight-alpha image
    at dst size, or None if the native library is missing."""
    if im is None or not available():
        return None
    dst_w, dst_h = int(dst_w), int(dst_h)
    if dst_w <= 0 or dst_h <= 0:
        return None
    from PIL import Image
    src = im.convert("RGBA")
    sw, sh = src.size
    raw = premultiply(src.tobytes(), sw, sh)
    out = scale_rgba(raw, sw, sh, dst_w, dst_h, filter)
    if out is None:
        return None
    straight = unpremultiply(out, dst_w, dst_h)
    return Image.frombytes("RGBA", (dst_w, dst_h), straight)


def _cache_get(key):
    return _cache.get(key)


def _cache_put(key, value):
    global _cache
    gen = _cache_gen
    cur = _cache
    if cur.get(key) is value:
        return
    new = dict(cur)
    if key not in new and len(new) >= _CACHE_MAX:
        # Drop an arbitrary oldest-inserted key; insertion order
        # is enough here (CPython 3.7+).
        try:
            new.pop(next(iter(new)))
        except StopIteration:
            pass
    new[key] = value
    if _cache_gen == gen:
        _cache = new


def clear_cache():
    global _cache, _cache_gen
    _cache_gen += 1
    _cache = {}


def cached_scale_image(im, dst_w, dst_h, filter="lanczos3", key=None):
    """Like scale_image, with an optional cache key (path, mtime, …)."""
    filt = _norm_filter(filter)
    dst_w, dst_h = int(dst_w), int(dst_h)
    if key is not None:
        ck = (key, dst_w, dst_h, filt)
        hit = _cache_get(ck)
        if hit is not None:
            return hit
    out = scale_image(im, dst_w, dst_h, filt)
    if out is not None and key is not None:
        _cache_put(ck, out)
    return out


def load_sized(paths_by_size, dest, filter="lanczos3"):
    """Open a raster asset at `dest` pixels.

    paths_by_size maps int size -> file path (a square image). Picks
    the nearest larger prebuilt size and, if the native scaler is
    loaded, resamples it to exactly `dest`. If the scaler is missing,
    returns the nearest prebuilt image unchanged (no Python resample).
    Returns a Pillow RGBA image or None.
    """
    if not paths_by_size:
        return None
    sizes = []
    table = {}
    for k, path in paths_by_size.items():
        try:
            sz = int(k)
        except (TypeError, ValueError):
            continue
        if path and os.path.isfile(path):
            sizes.append(sz)
            table[sz] = path
    if not sizes:
        return None
    dest = int(dest)
    src_sz = pick_source_size(sizes, dest)
    path = table.get(src_sz)
    if not path:
        return None
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    if im.size == (dest, dest) or not available():
        # Missing DLL: hand back the prebuilt size as-is, even if it
        # is not dest. Identity is also a no-resample path.
        return im
    key = (os.path.abspath(path), os.path.getmtime(path), im.size)
    out = cached_scale_image(im, dest, dest, filter, key=key)
    return out if out is not None else im


def load_path_scaled(path, dest_w, dest_h, filter="lanczos3"):
    """Scale a single file to dest_w x dest_h, or return it as-is
    (no Python resample) when the library is missing."""
    if not path or not os.path.isfile(path):
        return None
    from PIL import Image
    im = Image.open(path).convert("RGBA")
    dest_w, dest_h = int(dest_w), int(dest_h)
    if im.size == (dest_w, dest_h) or not available():
        return im
    key = (os.path.abspath(path), os.path.getmtime(path), im.size)
    out = cached_scale_image(im, dest_w, dest_h, filter, key=key)
    return out if out is not None else im
