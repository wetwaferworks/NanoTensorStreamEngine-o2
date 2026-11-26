#!/usr/bin/env python3
"""nemo_icons.py - model-icon registry, pack loader, and the
thinking-line animation state machine (no Tk).

A model can show either a painted icon pack (PNG frame strip or a
folder of numbered frames, described by ICONS.md) or a procedural
thinking shape (nemo_shapes.py). A valid pack in icon= wins; a
missing or bad pack falls back to the shape, then to the built-in
default. Users add their own pack by pointing icon= at it.

Animation states are driven by the existing engine protocol events
and are computed here so the GUI only has to paint: idle, thinking,
streaming, expanding (EXPAND), dtc (DTC), error, done. Motion (flip, bob,
wobble, spin, pulse) is a small transform on top of the current
frame, so a one-frame pack still has personality.
"""

import json
import os
import time

import nemo_shapes as shapes

STATES = ("idle", "thinking", "streaming", "expanding", "dtc", "error", "done")

# Distinct default motions per state. The model's configured motion
# is used for thinking; expanding always uses a different one so the
# DTC / expansion work is visually distinct.
_STATE_MOTION = {
    "idle": None,
    "thinking": None,          # filled in from the model
    "streaming": "wobble",
    "expanding": "spin",
    "dtc": "pulse",
    "error": "wobble",
    "done": "pulse",
}

# How long done/error keep a finishing flourish before they settle.
SETTLE_S = {"done": 0.9, "error": 0.6, "idle": 0.0}

# Default fps for procedural animation (packs may override per state).
DEFAULT_FPS = 12


def bundled_icons_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "icons")


def bundled_themes_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")


# ---------------------------------------------------------- motion

def motion_transform(motion, phase, size=32):
    """phase is radians (already includes speed). Returns a transform
    dict for nemo_shapes / the painter. Amplitude scales with size
    so 32 and 64 px icons travel the same relative distance."""
    import math

    if not motion or motion not in shapes.MOTIONS:
        return {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0,
                "flip_x": 1.0, "squash": 1.0}
    amp = size * 0.07
    if motion == "bob":
        return {"dx": 0.0, "dy": math.sin(phase) * amp,
                "scale": 1.0, "rot": 0.0, "flip_x": 1.0,
                "squash": 1.0 + 0.04 * math.sin(phase * 2)}
    if motion == "wobble":
        return {"dx": math.sin(phase) * amp * 0.4,
                "dy": 0.0, "scale": 1.0,
                "rot": math.sin(phase) * 14.0,
                "flip_x": 1.0, "squash": 1.0}
    if motion == "spin":
        return {"dx": 0.0, "dy": 0.0, "scale": 1.0,
                "rot": (phase * 180.0 / math.pi) % 360.0,
                "flip_x": 1.0, "squash": 1.0}
    if motion == "pulse":
        s = 1.0 + 0.12 * math.sin(phase)
        return {"dx": 0.0, "dy": 0.0, "scale": s, "rot": 0.0,
                "flip_x": 1.0, "squash": 1.0}
    # flip: a card-flip on the vertical axis, with a little squash
    # at the thin point so it reads as a flip, not a disappear.
    fx = math.cos(phase)
    squash = 1.0 + 0.12 * (1.0 - abs(fx))
    return {"dx": 0.0, "dy": 0.0, "scale": 1.0, "rot": 0.0,
            "flip_x": fx if abs(fx) > 0.06 else (0.06 if fx >= 0 else -0.06),
            "squash": squash}


def motion_for(state, preferred):
    if state == "thinking":
        return preferred if preferred in shapes.MOTIONS else "bob"
    motion = _STATE_MOTION.get(state)
    if state == "expanding" and preferred == "spin":
        return "pulse"          # keep expanding visually distinct
    if state == "dtc" and preferred == "pulse":
        return "spin"
    if state == "streaming" and preferred == "wobble":
        return "bob"
    return motion


# ---------------------------------------------------------- packs

class IconPack:
    """A loaded (or built-in) icon pack. frames[size] is a list of
    Pillow images; states maps state -> {frames, fps, loop, motion}."""

    def __init__(self, ident, name, source, frames, states, fps=DEFAULT_FPS,
                 fallback=False, reason="", apply_motion=True, theme=""):
        self.id = ident
        self.name = name
        self.source = source
        self.frames = frames          # {32: [Image, ...], 64: [...]}
        self.states = states
        self.fps = fps
        self.fallback = fallback
        self.reason = reason
        # Official mascots paint motion into the frames; applying a
        # second transform would stack (wobble-on-wobble, etc.).
        self.apply_motion = apply_motion
        self.theme = theme or ""

    def state_frames(self, state):
        info = self.states.get(state) or self.states.get("idle") or {}
        idx = list(info.get("frames") or [0])
        return idx, float(info.get("fps") or self.fps), bool(info.get("loop", True))

    def image(self, state, index, size=32):
        """The closest prebuilt frame for `size`. Does not resample;
        the caller scales with the C library if it needs an exact size."""
        size = int(size)
        sizes = [sz for sz, bank in (self.frames or {}).items() if bank]
        if not sizes:
            return None
        import nemo_scale as _sc
        pick = _sc.pick_source_size(sizes, size)
        bank = self.frames.get(pick) or self.frames.get(size) or []
        if not bank:
            bank = self.frames[sizes[0]]
        idx, _, _ = self.state_frames(state)
        if not idx:
            return bank[0]
        return bank[idx[index % len(idx)] % len(bank)]


def _default_states(n, fps=DEFAULT_FPS):
    """Spread n frames across the six states when a pack has no
    manifest (a bare strip or a folder of numbered PNGs)."""
    n = max(1, int(n))
    # First frame is idle/done/error; the rest cycle for motion states.
    idle = [0]
    rest = list(range(n)) if n == 1 else list(range(1, n)) or [0]
    return {
        "idle": {"frames": idle, "fps": 1, "loop": True},
        "thinking": {"frames": rest, "fps": fps, "loop": True},
        "streaming": {"frames": rest, "fps": fps, "loop": True},
        "expanding": {"frames": list(reversed(rest)), "fps": max(6, fps - 2),
                      "loop": True},
        "dtc": {"frames": list(reversed(rest)), "fps": max(6, fps),
                "loop": True, "motion": "pulse"},
        "error": {"frames": idle, "fps": 1, "loop": False},
        "done": {"frames": idle, "fps": 1, "loop": False},
    }


def _open_rgba(path):
    from PIL import Image

    im = Image.open(path)
    im.load()
    return im.convert("RGBA")


def _split_strip(im, frame_size=None):
    """Horizontal (or vertical, if taller than wide) strip of square
    frames. frame_size, when given, must divide the long side."""
    w, h = im.size
    if frame_size:
        side = int(frame_size)
        if side <= 0 or (w >= h and w % side) or (h > w and h % side):
            raise ValueError(f"strip {w}x{h} is not a multiple of {side}")
    else:
        side = min(w, h)
        if side <= 0 or max(w, h) % side:
            raise ValueError(f"strip {w}x{h} is not a run of square frames")
    frames = []
    if w >= h:
        for x in range(0, w, side):
            frames.append(im.crop((x, 0, x + side, side)))
    else:
        for y in range(0, h, side):
            frames.append(im.crop((0, y, side, y + side)))
    if not frames:
        raise ValueError("strip contained no frames")
    return frames


def _numbered_pngs(folder):
    names = sorted(n for n in os.listdir(folder)
                   if n.lower().endswith(".png"))
    if not names:
        raise ValueError("no PNG frames in the folder")
    return [_open_rgba(os.path.join(folder, n)) for n in names]


def _rescale_bank(frames, size, filter="lanczos3"):
    """Scale a frame list to `size` with the C scaler. Returns None
    if the library is missing — callers must then keep the native
    size (no Python resampling)."""
    import nemo_scale as _sc

    out = []
    for im in frames:
        if im.size == (size, size):
            out.append(im)
            continue
        scaled = _sc.scale_image(im, size, size, filter)
        if scaled is None:
            return None
        out.append(scaled)
    return out


def _validate_states(raw, nframes):
    states = {}
    if not isinstance(raw, dict):
        raise ValueError("manifest 'states' must be an object")
    for name, info in raw.items():
        if name not in STATES:
            continue
        if not isinstance(info, dict):
            raise ValueError(f"state {name!r} must be an object")
        frames = info.get("frames")
        if not frames:
            raise ValueError(f"state {name!r} has no frames")
        idx = []
        for i in frames:
            try:
                i = int(i)
            except (TypeError, ValueError):
                raise ValueError(f"state {name!r} has a non-integer frame")
            if i < 0 or i >= nframes:
                raise ValueError(f"state {name!r} frame {i} is out of range "
                                 f"(pack has {nframes} frames)")
            idx.append(i)
        fps = info.get("fps")
        try:
            fps = float(fps) if fps is not None else DEFAULT_FPS
        except (TypeError, ValueError):
            raise ValueError(f"state {name!r} fps is not a number")
        if fps <= 0 or fps > 60:
            raise ValueError(f"state {name!r} fps must be in (0, 60]")
        motion = info.get("motion")
        if motion and motion not in shapes.MOTIONS:
            raise ValueError(f"state {name!r} motion must be one of "
                             + ", ".join(shapes.MOTIONS))
        states[name] = {
            "frames": idx, "fps": fps,
            "loop": bool(info.get("loop", True)),
            "motion": motion,
        }
    if "idle" not in states:
        states["idle"] = {"frames": [0], "fps": 1, "loop": True}
    for name in STATES:
        states.setdefault(name, states["idle"])
    return states



def alpha_blit(dst, src, xy=(0, 0)):
    """Composite `src` over `dst` with proper partial-alpha blending.
    Pillow's paste(..., mask=src) treats the mask more like a cutout
    and drops in-between alpha; this uses alpha_composite instead."""
    from PIL import Image

    if dst.mode != "RGBA":
        dst = dst.convert("RGBA")
    if src.mode != "RGBA":
        src = src.convert("RGBA")
    ox, oy = int(xy[0]), int(xy[1])
    sw, sh = src.size
    dw, dh = dst.size
    dx0, dy0 = max(ox, 0), max(oy, 0)
    sx0, sy0 = dx0 - ox, dy0 - oy
    dx1, dy1 = min(ox + sw, dw), min(oy + sh, dh)
    if dx1 <= dx0 or dy1 <= dy0:
        return dst
    sx1, sy1 = sx0 + (dx1 - dx0), sy0 + (dy1 - dy0)
    crop = src if (sx0, sy0, sx1, sy1) == (0, 0, sw, sh) else src.crop(
        (sx0, sy0, sx1, sy1))
    layer = Image.new("RGBA", dst.size, (0, 0, 0, 0))
    layer.paste(crop, (dx0, dy0))
    return Image.alpha_composite(dst, layer)


def _is_mascot_v1(manifest):
    """True for CloudAtlas ntse-mascot/1 packs (per-state strips,
    `frames` as a count, extra keys such as fallback_state)."""
    if not isinstance(manifest, dict):
        return False
    schema = str(manifest.get("schema") or "")
    if schema.startswith("ntse-mascot/"):
        return True
    states = manifest.get("states")
    if not isinstance(states, dict):
        return False
    for info in states.values():
        if not isinstance(info, dict):
            continue
        if isinstance(info.get("strip"), dict):
            return True
        if isinstance(info.get("frames"), int):
            return True
    return False


def _frame_side(manifest, sz):
    fs = manifest.get("frame_size")
    if isinstance(fs, dict):
        v = fs.get(str(sz), fs.get(sz))
        if isinstance(v, (list, tuple)) and v:
            return int(v[0])
        if isinstance(v, (int, float)):
            return int(v)
    if isinstance(fs, (int, float)):
        return int(fs)
    return int(sz)


def _state_frame_count(info):
    raw = info.get("frames")
    if isinstance(raw, int):
        if raw < 1:
            raise ValueError("state frame count must be at least 1")
        return raw, None
    if isinstance(raw, list):
        if not raw:
            raise ValueError("state has no frames")
        idx = []
        for i in raw:
            try:
                i = int(i)
            except (TypeError, ValueError):
                raise ValueError("state has a non-integer frame")
            if i < 0:
                raise ValueError("state frame index is negative")
            idx.append(i)
        return (max(idx) + 1 if idx else 0), idx
    if raw in (None, "", 0):
        return None, None
    raise ValueError("state frames must be a count or a list of indices")


def _scale_filter(manifest):
    scaling = str(manifest.get("scaling") or "").strip().lower()
    if scaling in ("nearest", "nearest-integer", "nearest-neighbour",
                   "nearest-neighbor"):
        return "nearest"
    return "lanczos3"


def _load_state_strip(folder, manifest, name, info, sz):
    """One state's frames at `sz`, or None if that size is missing."""
    candidates = []
    strip = info.get("strip") if isinstance(info.get("strip"), dict) else {}
    rel = strip.get(str(sz), strip.get(sz) if not isinstance(sz, str) else None)
    if rel:
        candidates.append(os.path.join(folder, rel))
    candidates.append(os.path.join(folder, f"{name}-{sz}.png"))
    # Numbered frames: <state>-<size>-00.png
    numbered = []
    prefix = f"{name}-{sz}-"
    try:
        names = os.listdir(folder)
    except OSError:
        names = []
    for n in sorted(names):
        low = n.lower()
        if low.startswith(prefix.lower()) and low.endswith(".png"):
            numbered.append(os.path.join(folder, n))
    last_err = None
    for cand in candidates:
        if not os.path.isfile(cand):
            continue
        try:
            return _split_strip(_open_rgba(cand), _frame_side(manifest, sz))
        except ValueError as exc:
            last_err = exc
    if numbered:
        return [_open_rgba(n) for n in numbered]
    if last_err:
        raise last_err
    return None


def _load_mascot_v1(folder, manifest):
    """Load an ntse-mascot/1 pack: per-state horizontal strips, a
    frame count (or index list) per state, and optional extra keys
    (loop, fallback_state, description). Missing states use
    fallback_state (default idle). Repeated motion tags are allowed
    — the same motion may appear on more than one state."""
    raw_states = manifest.get("states")
    if not isinstance(raw_states, dict):
        raise ValueError("manifest 'states' must be an object")

    sizes = manifest.get("sizes") or [32, 64]
    try:
        sizes = [int(s) for s in sizes]
    except (TypeError, ValueError):
        raise ValueError("manifest sizes must be integers")
    if not sizes:
        sizes = [32, 64]
    if 32 not in sizes:
        sizes = [32] + sizes

    fallback = (manifest.get("fallback_state") or "idle").strip() or "idle"
    if fallback not in STATES:
        fallback = "idle"

    # Parse every known state that is present; unknown keys are ignored.
    parsed = {}
    for name, info in raw_states.items():
        if name not in STATES:
            continue
        if not isinstance(info, dict):
            raise ValueError(f"state {name!r} must be an object")
        count, idx = _state_frame_count(info)
        fps = info.get("fps")
        try:
            fps = float(fps) if fps is not None else DEFAULT_FPS
        except (TypeError, ValueError):
            raise ValueError(f"state {name!r} fps is not a number")
        if fps <= 0 or fps > 60:
            raise ValueError(f"state {name!r} fps must be in (0, 60]")
        motion = info.get("motion")
        if motion and motion not in shapes.MOTIONS:
            raise ValueError(f"state {name!r} motion must be one of "
                             + ", ".join(shapes.MOTIONS))
        # description / loop / any other extra keys are accepted.
        parsed[name] = {
            "count": count,
            "idx": idx,
            "fps": fps,
            "loop": bool(info.get("loop", True)),
            "motion": motion,
            "description": info.get("description") or "",
            "info": info,
        }

    if "idle" not in parsed:
        raise ValueError("ntse-mascot/1 pack must include an idle state")

    banks = {sz: [] for sz in (32, 64)}
    states = {}
    offsets = {}
    filt = _scale_filter(manifest)

    def append_state(name, rec):
        info = rec["info"]
        loaded = {}
        for sz in sizes:
            try:
                frames = _load_state_strip(folder, manifest, name, info, sz)
            except ValueError:
                frames = None
            if frames:
                loaded[sz] = frames
        if 32 not in loaded and 64 not in loaded:
            raise ValueError(f"state {name!r} has no usable frames")
        native = loaded.get(32) or loaded.get(64)
        for sz in (32, 64):
            if sz in loaded:
                continue
            scaled = _rescale_bank(native, sz, filter=filt)
            if scaled is None:
                # Keep going with whatever sizes we have; IconPack.image
                # picks the closest bank.
                continue
            loaded[sz] = scaled
        n_native = len(next(iter(loaded.values())))
        count = rec["count"]
        if count is not None and count != n_native:
            raise ValueError(
                f"state {name!r} strip has {n_native} frames, "
                f"manifest says {count}")
        idx = rec["idx"]
        if idx is None:
            idx = list(range(n_native))
        else:
            for i in idx:
                if i >= n_native:
                    raise ValueError(
                        f"state {name!r} frame {i} is out of range "
                        f"(state has {n_native} frames)")
        offset = len(next(iter(banks.values()))) if any(banks.values()) else 0
        # Keep every size bank the same length.
        for sz in (32, 64):
            bank = loaded.get(sz)
            if bank is None:
                # Duplicate the closest bank's count with empty? No —
                # scale from any loaded size if the earlier pass missed.
                src = loaded.get(32) or loaded.get(64)
                bank = _rescale_bank(src, sz, filter=filt)
                if bank is None:
                    raise ValueError(
                        f"state {name!r} could not provide {sz}px frames")
            banks[sz].extend(bank)
        offsets[name] = offset
        states[name] = {
            "frames": [offset + i for i in idx],
            "fps": rec["fps"],
            "loop": rec["loop"],
            "motion": rec["motion"],
            "description": rec["description"],
        }

    # Load idle first so fallback can share its indices.
    append_state("idle", parsed["idle"])
    for name in STATES:
        if name == "idle":
            continue
        if name in parsed:
            append_state(name, parsed[name])

    # Missing states share the fallback state's frame indices.
    fb = fallback if fallback in states else "idle"
    for name in STATES:
        if name not in states:
            src = states[fb]
            states[name] = {
                "frames": list(src["frames"]),
                "fps": src["fps"],
                "loop": src["loop"],
                "motion": src.get("motion"),
                "description": src.get("description") or "",
            }

    name = (manifest.get("name") or os.path.basename(folder)).strip() or "icon"
    ident = os.path.abspath(folder)
    fps = manifest.get("fps") or DEFAULT_FPS
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        fps = DEFAULT_FPS
    return IconPack(
        ident=ident, name=name, source=ident, frames=banks,
        states=states, fps=fps, apply_motion=False,
        theme=(manifest.get("theme") or "").strip())


# Official CloudAtlas mascots, one pack per theme. Pixel Nemo is the default.
THEME_MASCOT_IDS = {
    "pixel": "pixel-nemo",
    "lattice": "lattice-cube",
    "stream": "stream-critter",
    "handshake": "handshake-oO",
    "owl": "sage-owl",
}


def theme_mascot_pack(theme_id, search_dirs=None):
    """The official mascot pack for a theme id, or None."""
    theme_id = (theme_id or "").strip()
    if not theme_id:
        return None
    want = THEME_MASCOT_IDS.get(theme_id, theme_id)
    dirs = []
    for d in list(search_dirs or []) + [bundled_icons_dir()]:
        if d and d not in dirs:
            dirs.append(d)
    candidates = []
    for folder in dirs:
        if not folder or not os.path.isdir(folder):
            continue
        for name in (want, theme_id):
            path = os.path.join(folder, name)
            if os.path.isfile(os.path.join(path, "manifest.json")):
                candidates.append(path)
        try:
            names = os.listdir(folder)
        except OSError:
            continue
        for name in names:
            path = os.path.join(folder, name)
            man = os.path.join(path, "manifest.json")
            if not os.path.isfile(man):
                continue
            try:
                with open(man, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict):
                continue
            if data.get("theme") == theme_id or data.get("id") == want:
                candidates.append(path)
    seen = set()
    for path in candidates:
        ap = os.path.abspath(path)
        if ap in seen:
            continue
        seen.add(ap)
        try:
            return load_pack_from_path(ap)
        except ValueError:
            continue
    return None


def load_pack_from_path(path):
    """Load a pack from a folder (manifest + strip or frames) or from
    a bare PNG strip. Raises ValueError with a reason on any problem;
    the registry turns that into a fallback."""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise ValueError(f"icon pack not found: {path}")
    if os.path.isfile(path):
        im = _open_rgba(path)
        frames = _split_strip(im)
        banks = {im.height if im.width >= im.height else im.width: frames}
        # Offer both sizes when we can scale.
        side = next(iter(banks))
        for sz in (32, 64):
            if sz not in banks:
                scaled = _rescale_bank(frames, sz)
                if scaled is not None:
                    banks[sz] = scaled
        return IconPack(ident=path, name=os.path.splitext(
            os.path.basename(path))[0], source=path, frames=banks,
                        states=_default_states(len(frames)),
                        fps=DEFAULT_FPS)

    manifest_path = os.path.join(path, "manifest.json")
    manifest = {}
    if os.path.isfile(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"manifest.json is not valid JSON: {exc}")
        if not isinstance(manifest, dict):
            raise ValueError("manifest.json must be an object")

    if _is_mascot_v1(manifest):
        return _load_mascot_v1(path, manifest)

    sizes = manifest.get("sizes") or [32, 64]
    try:
        sizes = [int(s) for s in sizes]
    except (TypeError, ValueError):
        raise ValueError("manifest sizes must be integers")
    if not sizes:
        sizes = [32, 64]

    banks = {}
    fmt = (manifest.get("format") or "").strip().lower()
    strip_map = manifest.get("strip") or {}
    frames_map = manifest.get("frames") or {}

    def load_size(sz):
        # Prefer an explicit path in the manifest, then conventions.
        candidates = []
        if isinstance(strip_map, dict) and str(sz) in strip_map:
            candidates.append(("strip", os.path.join(path, strip_map[str(sz)])))
        if isinstance(frames_map, dict) and str(sz) in frames_map:
            candidates.append(("dir", os.path.join(path, frames_map[str(sz)])))
        candidates.extend([
            ("strip", os.path.join(path, f"strip-{sz}.png")),
            ("dir", os.path.join(path, f"frames-{sz}")),
        ])
        if fmt == "frames":
            candidates = [c for c in candidates if c[0] == "dir"] + \
                         [c for c in candidates if c[0] == "strip"]
        last_err = None
        for kind, cand in candidates:
            try:
                if kind == "strip" and os.path.isfile(cand):
                    return _split_strip(_open_rgba(cand),
                                        manifest.get("frame_size") or sz)
                if kind == "dir" and os.path.isdir(cand):
                    return _numbered_pngs(cand)
            except ValueError as exc:
                last_err = exc
        if last_err:
            raise last_err
        raise ValueError(f"no {sz}px frames (looked for strip-{sz}.png or "
                         f"frames-{sz}/)")

    # Load the native sizes we can; scale the rest.
    native = None
    for sz in sizes:
        try:
            banks[sz] = load_size(sz)
            native = banks[sz]
        except ValueError:
            continue
    if native is None:
        # Last chance: a single strip.png of unknown size.
        bare = os.path.join(path, "strip.png")
        if os.path.isfile(bare):
            native = _split_strip(_open_rgba(bare))
            side = native[0].size[0]
            banks[side] = native
        else:
            raise ValueError("no usable frames in the pack")
    for sz in (32, 64):
        if sz not in banks:
            scaled = _rescale_bank(native, sz)
            if scaled is not None:
                banks[sz] = scaled

    nframes = len(next(iter(banks.values())))
    if nframes < 1:
        raise ValueError("pack has no frames")
    if "states" in manifest:
        states = _validate_states(manifest["states"], nframes)
    else:
        states = _default_states(nframes, manifest.get("fps") or DEFAULT_FPS)
    fps = manifest.get("fps") or DEFAULT_FPS
    try:
        fps = float(fps)
    except (TypeError, ValueError):
        raise ValueError("manifest fps is not a number")
    name = (manifest.get("name") or os.path.basename(path)).strip() or "icon"
    return IconPack(ident=path, name=name, source=path, frames=banks,
                    states=states, fps=fps)


def _builtin_pack(model_key):
    """An in-memory pack rendered from the model's default shape, so
    a missing file can never leave the GUI without an icon."""
    cfg = shapes.DEFAULT_SHAPES.get(model_key) or shapes.DEFAULT_SHAPES["default"]
    frames = {32: [], 64: []}
    # 8 baked poses: idle, look-up, squash, tilt-ish via phase, etc.
    # States pick from these; motion is still applied on top.
    phases = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    pose_state = ["idle", "thinking", "thinking", "streaming",
                  "expanding", "expanding", "error", "done"]
    for i, (ph, st) in enumerate(zip(phases, pose_state)):
        for sz in (32, 64):
            frames[sz].append(shapes.render_pil(cfg, state=st, size=sz,
                                                phase=ph))
    states = {
        "idle": {"frames": [0], "fps": 1, "loop": True},
        "thinking": {"frames": [1, 2, 3, 4], "fps": 8, "loop": True},
        "streaming": {"frames": [3, 4, 5], "fps": 10, "loop": True},
        "expanding": {"frames": [4, 5, 1, 2], "fps": 8, "loop": True},
        "dtc": {"frames": [5, 4, 2, 1], "fps": 8, "loop": True,
                "motion": "pulse"},
        "error": {"frames": [6], "fps": 1, "loop": False},
        "done": {"frames": [7], "fps": 1, "loop": False},
    }
    names = {"supernemo": "SuperNemo (built-in)",
             "littlenemo": "Little Nemo (built-in)"}
    return IconPack(ident=f"builtin:{model_key}",
                    name=names.get(model_key, "Built-in"),
                    source="builtin", frames=frames, states=states,
                    fps=8, fallback=False)


_BUILTIN_CACHE = {}


def builtin_pack(model_key="default"):
    key = model_key if model_key in shapes.DEFAULT_SHAPES else "default"
    if key not in _BUILTIN_CACHE:
        _BUILTIN_CACHE[key] = _builtin_pack(key)
    return _BUILTIN_CACHE[key]


def discover_packs(search_dirs):
    """Packs that can be picked: built-ins, then any folder with a
    manifest.json (or a strip-32.png) under each search dir."""
    found = []
    seen = set()
    for key in ("supernemo", "littlenemo", "default"):
        p = builtin_pack(key)
        found.append({"id": p.id, "name": p.name, "source": "builtin",
                      "fallback": False})
        seen.add(p.id)
    for folder in search_dirs or []:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        # A pack is a subfolder, or a .png strip sitting in the folder.
        for name in names:
            path = os.path.join(folder, name)
            ident = os.path.abspath(path)
            if ident in seen:
                continue
            ok = False
            if os.path.isdir(path) and (
                    os.path.isfile(os.path.join(path, "manifest.json"))
                    or os.path.isfile(os.path.join(path, "strip-32.png"))
                    or os.path.isfile(os.path.join(path, "strip.png"))):
                ok = True
            elif os.path.isfile(path) and name.lower().endswith(".png"):
                ok = True
            if not ok:
                continue
            label = name
            manifest = os.path.join(path, "manifest.json") if os.path.isdir(path) else ""
            if manifest and os.path.isfile(manifest):
                try:
                    with open(manifest, "r", encoding="utf-8") as f:
                        label = json.load(f).get("name") or name
                except (OSError, json.JSONDecodeError, AttributeError):
                    pass
            found.append({"id": ident, "name": str(label), "source": ident,
                          "fallback": False})
            seen.add(ident)
    return found


def resolve_icon(spec, model_key="default", search_dirs=None):
    """spec (icon=) -> (IconPack, used_procedural: bool).
    A valid pack is returned with used_procedural=False. Anything else
    (empty spec, missing file, bad pack) returns the built-in pack and
    used_procedural=True so the GUI draws the procedural shape instead.
    The pack is still provided so a painter can fall back to its frames
    if it wants a bitmap."""
    spec = (spec or "").strip()
    if not spec:
        pack = builtin_pack(model_key)
        pack = IconPack(pack.id, pack.name, pack.source, pack.frames,
                        pack.states, pack.fps, fallback=True,
                        reason="no icon set; using the thinking shape")
        return pack, True
    if spec.startswith("builtin:"):
        key = spec.split(":", 1)[1] or model_key
        return builtin_pack(key), False
    if spec.startswith("theme-strip:"):
        tid = spec.split(":", 1)[1].strip()
        try:
            import nemo_themes as _themes
            theme = _themes.get_theme(tid, search_dirs)
            pack = _themes.thinking_icon_pack(theme) if theme else None
        except Exception as exc:
            pack = None
            last = str(exc)
        else:
            last = "theme thinking strip not found"
        if pack is not None:
            return pack, False
        pack = builtin_pack(model_key)
        pack = IconPack(pack.id, pack.name, pack.source, pack.frames,
                        pack.states, pack.fps, fallback=True, reason=last)
        return pack, True
    # Relative paths resolve against each search dir, then cwd.
    candidates = [spec] if os.path.isabs(spec) else []
    if not os.path.isabs(spec):
        for folder in list(search_dirs or []) + [os.getcwd()]:
            candidates.append(os.path.join(folder, spec))
    last = "icon pack not found"
    for cand in candidates:
        if os.path.exists(cand):
            try:
                return load_pack_from_path(cand), False
            except ValueError as exc:
                last = str(exc)
                break
    pack = builtin_pack(model_key)
    pack = IconPack(pack.id, pack.name, pack.source, pack.frames,
                    pack.states, pack.fps, fallback=True, reason=last)
    return pack, True


# ---------------------------------------------------------- state machine

def next_state(current, event, payload=None):
    """One protocol (or GUI) event -> the next animation state."""
    current = current if current in STATES else "idle"
    payload = payload or {}
    if event in ("turn_begin", "think"):
        return "thinking"
    if event == "tok":
        return "streaming"
    if event == "expand":
        if payload.get("state") == "end":
            return current if current in ("thinking", "streaming", "dtc") else "thinking"
        return "expanding"
    if event == "dtc":
        if payload.get("state") == "end":
            return current if current in ("thinking", "streaming", "expanding") else "thinking"
        return "dtc"
    if event == "error":
        return "error"
    if event in ("done", "turn_done"):
        return "error" if current == "error" else "done"
    if event in ("idle", "turn_idle"):
        return "idle"
    return current


class IconAnimator:
    """Advances frames and motion for one icon. Feed it events with
    on_event(); step(now) returns (state, frame_index, transform,
    settled). A fake clock makes this unit-testable. No threads, no
    locks: the GUI calls step() from an after() timer."""

    def __init__(self, cfg=None, model_key="default", pack=None,
                 clock=time.monotonic):
        self.clock = clock
        self.model_key = model_key
        self.configure(cfg, pack)
        self.state = "idle"
        self.frame = 0
        self.t0 = self.clock()
        self.t_state = self.t0
        self._phase0 = 0.0
        self.settled = True

    def configure(self, cfg=None, pack=None):
        self.cfg = shapes.normalize_shape(cfg, self.model_key)
        self.pack = pack
        self.procedural = pack is None or pack.fallback or not pack.frames

    def set_state(self, state, now=None):
        state = state if state in STATES else "idle"
        now = self.clock() if now is None else now
        if state != self.state:
            self.state = state
            self.frame = 0
            self.t_state = now
        self.settled = state in ("idle",) or (
            state in SETTLE_S and SETTLE_S[state] == 0)
        if state in ("thinking", "streaming", "expanding", "dtc"):
            self.settled = False

    def on_event(self, event, payload=None, now=None):
        self.set_state(next_state(self.state, event, payload), now=now)

    def interval_ms(self):
        if self.settled or self.state == "idle":
            return None
        fps = DEFAULT_FPS
        if self.pack and not self.procedural:
            _, fps, _ = self.pack.state_frames(self.state)
        fps = max(4.0, min(20.0, float(fps or DEFAULT_FPS)))
        return int(round(1000.0 / fps))

    def step(self, now=None):
        now = self.clock() if now is None else now
        elapsed = max(0.0, now - self.t_state)
        speed = self.cfg["speed"]
        # One full flip/bob cycle per 2/speed seconds at fps.
        import math
        phase = elapsed * speed * math.pi  # 1 Hz at speed=1 -> 2π/2
        preferred = self.cfg["motion"]
        motion = motion_for(self.state, preferred)
        if self.pack and not self.procedural:
            info = self.pack.states.get(self.state) or {}
            if not getattr(self.pack, "apply_motion", True):
                # Motion is already painted into the pack's frames.
                motion = None
            elif info.get("motion"):
                motion = info["motion"]
        if self.state in ("idle",) or (self.state in SETTLE_S and
                                       elapsed + 1e-9 >= SETTLE_S[self.state]):
            if self.state != "idle" and self.state in SETTLE_S:
                # Stay on the state's rest pose, no motion, and settle.
                motion = None
                self.settled = True
            elif self.state == "idle":
                motion = None
                self.settled = True
        xf = motion_transform(motion, phase, self.cfg["size"])
        # Frame index from pack fps, or a slow cycle for procedural.
        fps = DEFAULT_FPS
        loop = True
        n = 1
        if self.pack:
            idx, fps, loop = self.pack.state_frames(self.state)
            n = max(1, len(idx))
        if loop and not self.settled:
            self.frame = int(elapsed * fps) % n
        else:
            self.frame = min(n - 1, int(elapsed * fps))
        return self.state, self.frame, xf, self.settled


# ---------------------------------------------------------- bake packs

def write_builtin_packs(dest_dir=None):
    """Write the placeholder packs CloudAtlas will replace: 32 and 64
    px frame strips, numbered PNGs, and a manifest.json for each of
    SuperNemo (purple blob) and Little Nemo (orange droplet). Returns
    the pack folders written."""
    dest_dir = dest_dir or bundled_icons_dir()
    os.makedirs(dest_dir, exist_ok=True)
    written = []
    specs = (
        ("supernemo", "SuperNemo placeholder",
         "A purple thinking blob with an orange outline. Replace these "
         "PNGs in place; keep manifest.json's frame indices if you keep "
         "the same number of frames."),
        ("littlenemo", "Little Nemo placeholder",
         "An orange thinking droplet with a purple outline. Replace "
         "these PNGs in place; keep manifest.json's frame indices if "
         "you keep the same number of frames."),
    )
    for key, name, note in specs:
        folder = os.path.join(dest_dir, key)
        _write_one_pack(folder, key, name, note)
        written.append(folder)
    return written


def _write_one_pack(folder, model_key, name, note):
    from PIL import Image

    cfg = shapes.DEFAULT_SHAPES[model_key]
    # 8 frames, matching _builtin_pack so builtin: and the files agree.
    phases = [0.0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8]
    pose_state = ["idle", "thinking", "thinking", "streaming",
                  "expanding", "expanding", "error", "done"]
    os.makedirs(folder, exist_ok=True)
    for sz in (32, 64):
        frames_dir = os.path.join(folder, f"frames-{sz}")
        os.makedirs(frames_dir, exist_ok=True)
        bank = []
        for i, (ph, st) in enumerate(zip(phases, pose_state)):
            im = shapes.render_pil(cfg, state=st, size=sz, phase=ph)
            im.save(os.path.join(frames_dir, f"{i:03d}.png"))
            bank.append(im)
        strip = Image.new("RGBA", (sz * len(bank), sz), (0, 0, 0, 0))
        for i, im in enumerate(bank):
            strip.paste(im, (i * sz, 0))
        strip.save(os.path.join(folder, f"strip-{sz}.png"))
    manifest = {
        "name": name,
        "author": "NTSE placeholders (CloudAtlas will replace)",
        "note": note,
        "sizes": [32, 64],
        "fps": 8,
        "format": "strip",
        "frame_size": 32,
        "layout": "horizontal",
        "strip": {"32": "strip-32.png", "64": "strip-64.png"},
        "frames": {"32": "frames-32", "64": "frames-64"},
        "states": {
            "idle": {"frames": [0], "fps": 1, "loop": True},
            "thinking": {"frames": [1, 2, 3, 4], "fps": 8, "loop": True,
                         "motion": "bob" if model_key == "supernemo" else "flip"},
            "streaming": {"frames": [3, 4, 5], "fps": 10, "loop": True,
                          "motion": "wobble"},
            "expanding": {"frames": [4, 5, 1, 2], "fps": 8, "loop": True,
                          "motion": "spin"},
            "dtc": {"frames": [5, 4, 2, 1], "fps": 8, "loop": True,
                    "motion": "pulse"},
            "error": {"frames": [6], "fps": 1, "loop": False},
            "done": {"frames": [7], "fps": 1, "loop": False},
        },
    }
    # frame_size in the manifest applies when reading a strip; the 64
    # px strip has 64 px frames, so we omit a single frame_size for
    # mixed strips and let the loader use the size key instead.
    manifest.pop("frame_size")
    with open(os.path.join(folder, "manifest.json"), "w",
              encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    dest = write_builtin_packs()
    for d in dest:
        print("wrote", d)
