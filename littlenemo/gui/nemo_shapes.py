#!/usr/bin/env python3
"""nemo_shapes.py - procedural thinking-shape geometry (no Tk).

A thinking shape is a little personality for a model: one of six base
forms with a fill, an outline, an optional face, a size and a motion.
The GUI draws these on a Canvas; this module is the geometry, the
motion transforms, and a Pillow renderer used to bake placeholder
icon packs (see ICONS.md). CloudAtlas's painted packs replace these
when a model has a valid icon= path.

Everything here is deterministic and headless-testable.
"""

import math

SHAPES = ("blob", "orb", "cube", "star", "ring", "droplet")
MOTIONS = ("flip", "bob", "wobble", "spin", "pulse")
SIZES = (32, 64)
FACE_STATES = ("idle", "thinking", "streaming", "expanding", "dtc", "error", "done")

PURPLE = "#A970FF"
ORANGE = "#FF9B42"
INK = "#1A1228"
WHITE = "#F7F4FF"

# Per-model built-in personalities (used when the INI has no override).
DEFAULT_SHAPES = {
    "supernemo": {
        "shape": "blob", "fill": PURPLE, "outline": ORANGE,
        "face": True, "size": 32, "motion": "bob", "speed": 1.0, "icon": "",
    },
    "littlenemo": {
        "shape": "droplet", "fill": ORANGE, "outline": PURPLE,
        "face": True, "size": 32, "motion": "flip", "speed": 1.0, "icon": "",
    },
    "default": {
        "shape": "orb", "fill": PURPLE, "outline": ORANGE,
        "face": True, "size": 32, "motion": "pulse", "speed": 1.0, "icon": "",
    },
}


def parse_color(value, default=None):
    """'#RGB' / '#RRGGBB' -> '#RRGGBB', or default when the value is bad."""
    s = (value or "").strip()
    if len(s) == 4 and s[0] == "#" and all(c in "0123456789abcdefABCDEF"
                                            for c in s[1:]):
        return ("#" + "".join(c * 2 for c in s[1:])).upper()
    if len(s) == 7 and s[0] == "#" and all(c in "0123456789abcdefABCDEF"
                                           for c in s[1:]):
        return s.upper()
    return default


def clamp_speed(value, default=1.0):
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    if v != v or v <= 0:
        return default
    return max(0.25, min(3.0, v))


def clamp_size(value, default=32):
    try:
        v = int(value)
    except (TypeError, ValueError):
        return default
    return 64 if v >= 48 else 32


def normalize_shape(cfg, model_key="default"):
    """One model's shape fields -> a complete, valid dict. Unknown or
    empty values fall back to that model's built-in personality, then
    to default. The original keys are kept so a later save is stable."""
    base = dict(DEFAULT_SHAPES.get(model_key) or DEFAULT_SHAPES["default"])
    raw = dict(base)
    if cfg:
        raw.update({k: cfg[k] for k in cfg})
    shape = (raw.get("shape") or base["shape"]).strip().lower()
    if shape not in SHAPES:
        shape = base["shape"]
    motion = (raw.get("motion") or base["motion"]).strip().lower()
    if motion not in MOTIONS:
        motion = base["motion"]
    fill = parse_color(raw.get("fill"), base["fill"])
    outline = parse_color(raw.get("outline"), base["outline"])
    face = raw.get("face")
    if isinstance(face, str):
        face = face.strip().lower() in ("1", "true", "yes", "on")
    else:
        face = bool(face) if face is not None else base["face"]
    return {
        "shape": shape,
        "fill": fill,
        "outline": outline,
        "face": bool(face),
        "size": clamp_size(raw.get("size"), base["size"]),
        "motion": motion,
        "speed": clamp_speed(raw.get("speed"), base["speed"]),
        "icon": (raw.get("icon") or "").strip(),
    }


# ---------------------------------------------------------- primitives

def _rot(x, y, cx, cy, xf):
    x, y = x - cx, y - cy
    x *= xf.get("flip_x", 1.0) * xf.get("scale", 1.0)
    y *= xf.get("squash", 1.0) * xf.get("scale", 1.0)
    rad = math.radians(xf.get("rot", 0.0))
    c, s = math.cos(rad), math.sin(rad)
    x, y = x * c - y * s, x * s + y * c
    return (x + cx + xf.get("dx", 0.0), y + cy + xf.get("dy", 0.0))


def _pts(points, cx, cy, xf):
    return [_rot(x, y, cx, cy, xf) for x, y in points]


def _circle(cx, cy, r, n=24):
    return [(cx + r * math.cos(2 * math.pi * i / n),
             cy + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


def blob_points(cx, cy, r, phase=0.0, n=20):
    """A friendly irregular blob; phase lets baked frames vary slightly."""
    out = []
    for i in range(n):
        a = 2 * math.pi * i / n - math.pi / 2
        wob = (0.16 * math.sin(3 * a + 0.5 + phase)
               + 0.07 * math.sin(5 * a - 0.4))
        rr = r * (1.0 + wob)
        out.append((cx + rr * math.cos(a), cy + rr * math.sin(a) * 1.04))
    return out


def star_points(cx, cy, r, points=5):
    out = []
    for i in range(points * 2):
        a = -math.pi / 2 + i * math.pi / points
        rr = r if i % 2 == 0 else r * 0.45
        out.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return out


def droplet_points(cx, cy, r, n=20):
    """A teardrop sitting on its point (thinking, head up)."""
    out = []
    for i in range(n):
        t = i / n
        a = -math.pi / 2 + t * 2 * math.pi
        # Stretch the bottom into a point, keep a round crown.
        if 0.15 < t < 0.85:
            rr = r * (0.55 + 0.45 * math.sin(t * math.pi))
        else:
            rr = r * (0.92 - 0.12 * math.cos(2 * a))
        # Pull the south pole down into a tip.
        y_pull = r * 0.35 * max(0.0, math.sin(a)) ** 2
        out.append((cx + rr * math.cos(a) * 0.86,
                    cy + rr * math.sin(a) + y_pull - r * 0.08))
    return out


def cube_faces(cx, cy, r):
    """Three isometric faces: top, left, right. Each is a 4-gon."""
    dx, dy = r * 0.70, r * 0.40
    top = (cx, cy - dy * 1.15)
    east = (cx + dx, cy - dy * 0.15)
    west = (cx - dx, cy - dy * 0.15)
    mid = (cx, cy + dy * 0.35)
    bot_e = (cx + dx, cy + dy * 1.05)
    bot_w = (cx - dx, cy + dy * 1.05)
    bot = (cx, cy + dy * 1.55)
    return {
        "top": [top, east, mid, west],
        "left": [west, mid, bot, bot_w],
        "right": [east, bot_e, bot, mid],
    }


def ring_radii(r):
    return r * 0.98, r * 0.46


# ---------------------------------------------------------- face

def face_layout(state, r):
    """Eye and mouth parameters for a state. Eyes look up while
    thinking, squint while expanding, frown on error, smile when
    done or streaming."""
    eye_y = -r * 0.10
    pupil_x, pupil_y = 0.0, 0.0
    eye_h = r * 0.22
    mouth = "smile"
    if state == "thinking":
        pupil_y = -r * 0.06
        eye_y = -r * 0.14
        mouth = "o"
    elif state == "streaming":
        pupil_x = r * 0.03
        mouth = "smile"
    elif state in ("expanding", "dtc"):
        eye_h = r * 0.13
        mouth = "line"
    elif state == "error":
        mouth = "frown"
        pupil_x = 0.0
    elif state == "done":
        mouth = "grin"
        eye_h = r * 0.10
        pupil_y = r * 0.01
    elif state == "idle":
        mouth = "smile"
    gap = r * 0.22
    return {
        "left": (-gap, eye_y), "right": (gap, eye_y),
        "eye_w": r * 0.18, "eye_h": eye_h,
        "pupil_w": r * 0.09, "pupil_h": r * 0.10,
        "pupil": (pupil_x, pupil_y),
        "mouth": mouth, "mouth_y": r * 0.28, "mouth_w": r * 0.22,
    }


def mouth_points(kind, cx, cy, w):
    if kind == "o":
        return ("oval", (cx - w * 0.35, cy - w * 0.28,
                         cx + w * 0.35, cy + w * 0.28))
    if kind == "line":
        return ("line", (cx - w * 0.5, cy, cx + w * 0.5, cy))
    # smile / grin / frown: a 3-point arc approximated as a polyline
    sign = -1 if kind == "frown" else 1
    depth = w * (0.55 if kind == "grin" else 0.40) * sign
    return ("poly", [(cx - w, cy), (cx, cy + depth), (cx + w, cy)])


# ---------------------------------------------------------- scene

def shape_prims(shape, cx, cy, r, phase=0.0):
    """Untransformed drawing primitives for a base shape.
    Each is ('poly'|'oval'|'line', points_or_bbox, role) where role is
    fill, fill2, fill3, or outline (outline is drawn last, as a stroke)."""
    shape = shape if shape in SHAPES else "orb"
    if shape == "blob":
        pts = blob_points(cx, cy, r, phase)
        return [("poly", pts, "fill"), ("poly", pts, "outline")]
    if shape == "orb":
        box = (cx - r, cy - r, cx + r, cy + r)
        return [("oval", box, "fill"), ("oval", box, "outline")]
    if shape == "star":
        pts = star_points(cx, cy, r)
        return [("poly", pts, "fill"), ("poly", pts, "outline")]
    if shape == "droplet":
        pts = droplet_points(cx, cy, r)
        return [("poly", pts, "fill"), ("poly", pts, "outline")]
    if shape == "ring":
        outer, inner = ring_radii(r)
        return [("oval", (cx - outer, cy - outer, cx + outer, cy + outer),
                 "fill"),
                ("oval", (cx - inner, cy - inner, cx + inner, cy + inner),
                 "hole"),
                ("oval", (cx - outer, cy - outer, cx + outer, cy + outer),
                 "outline"),
                ("oval", (cx - inner, cy - inner, cx + inner, cy + inner),
                 "outline")]
    # cube
    faces = cube_faces(cx, cy, r)
    return [("poly", faces["top"], "fill"),
            ("poly", faces["left"], "fill2"),
            ("poly", faces["right"], "fill3"),
            ("poly", faces["top"], "outline"),
            ("poly", faces["left"], "outline"),
            ("poly", faces["right"], "outline")]


def shade(hex_color, amount):
    """Lighten (amount>0) or darken a #RRGGBB color; amount in [-1, 1]."""
    c = parse_color(hex_color, PURPLE)
    r, g, b = int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)
    if amount >= 0:
        r, g, b = (int(r + (255 - r) * amount), int(g + (255 - g) * amount),
                   int(b + (255 - b) * amount))
    else:
        k = 1 + amount
        r, g, b = int(r * k), int(g * k), int(b * k)
    return f"#{max(0, min(255, r)):02X}{max(0, min(255, g)):02X}" \
           f"{max(0, min(255, b)):02X}"


def scene(cfg, state="idle", size=None, xf=None, phase=0.0):
    """A complete drawable scene: list of (kind, coords, fill, outline,
    width). Coordinates are in a square of `size` pixels, already
    transformed. Used by the Canvas painter and the Pillow renderer."""
    cfg = normalize_shape(cfg)
    size = int(size or cfg["size"])
    cx = cy = size / 2.0
    r = size * 0.36
    xf = dict(xf or {})
    prims = shape_prims(cfg["shape"], cx, cy, r, phase)
    fill, outline = cfg["fill"], cfg["outline"]
    colors = {
        "fill": (fill, ""),
        "fill2": (shade(fill, -0.18), ""),
        "fill3": (shade(fill, 0.12), ""),
        "hole": ("hole", ""),
        "outline": ("", outline),
    }
    out = []

    def xform_box(box):
        x0, y0, x1, y1 = box
        # Transform the four corners and take a new axis-aligned box
        # only for ovals that stay circular under uniform scale; for
        # flip/rot we convert the oval to a polygon so it rotates.
        corners = _pts([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], cx, cy, xf)
        if abs(xf.get("rot", 0.0)) < 0.5 and abs(xf.get("flip_x", 1.0)) == 1 \
                and abs(xf.get("squash", 1.0) - 1.0) < 0.02:
            xs, ys = zip(*corners)
            return ("oval", (min(xs), min(ys), max(xs), max(ys)))
        # midpoints of a 20-gon inscribed in the original oval
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        rx, ry = abs(x1 - x0) / 2, abs(y1 - y0) / 2
        pts = _pts([(mx + rx * math.cos(2 * math.pi * i / 20),
                     my + ry * math.sin(2 * math.pi * i / 20))
                    for i in range(20)], cx, cy, xf)
        return ("poly", pts)

    for kind, coords, role in prims:
        fill_c, out_c = colors[role]
        width = max(1.5, size / 22.0) if role == "outline" else 0
        if kind == "oval":
            k2, coords = xform_box(coords)
            out.append((k2, coords, fill_c, out_c, width))
        else:
            out.append((kind, _pts(coords, cx, cy, xf), fill_c, out_c, width))

    if cfg["face"]:
        face = face_layout(state, r)
        for side in ("left", "right"):
            ex, ey = face[side]
            ew, eh = face["eye_w"], face["eye_h"]
            box = (cx + ex - ew, cy + ey - eh, cx + ex + ew, cy + ey + eh)
            k2, coords = xform_box(box)
            out.append((k2, coords, WHITE, outline, max(1.0, size / 32.0)))
            px, py = face["pupil"]
            pw, ph = face["pupil_w"], face["pupil_h"]
            # X eyes on error: two short strokes instead of a pupil
            if state == "error":
                s = ew * 0.7
                ox, oy = cx + ex, cy + ey
                out.append(("line", _pts([(ox - s, oy - s), (ox + s, oy + s)],
                                         cx, cy, xf),
                            "", INK, max(1.2, size / 28.0)))
                out.append(("line", _pts([(ox - s, oy + s), (ox + s, oy - s)],
                                         cx, cy, xf),
                            "", INK, max(1.2, size / 28.0)))
            else:
                pbox = (cx + ex + px - pw, cy + ey + py - ph,
                        cx + ex + px + pw, cy + ey + py + ph)
                k2, coords = xform_box(pbox)
                out.append((k2, coords, INK, "", 0))
        mk, mcoords = mouth_points(face["mouth"], cx, cy + face["mouth_y"],
                                   face["mouth_w"])
        if mk == "oval":
            k2, coords = xform_box(mcoords)
            out.append((k2, coords, INK, "", 0))
        elif mk == "line":
            x0, y0, x1, y1 = mcoords
            out.append(("line", _pts([(x0, y0), (x1, y1)], cx, cy, xf),
                        "", INK, max(1.2, size / 26.0)))
        else:
            out.append(("line", _pts(mcoords, cx, cy, xf),
                        "", INK, max(1.4, size / 24.0)))
    return out


def render_pil(cfg, state="idle", size=32, xf=None, phase=0.0,
               background=None):
    """A RGBA Pillow image of the shape. Transparent if background is
    None; otherwise filled with that color first. Renders at 2x and
    downsamples so the edges stay soft."""
    from PIL import Image, ImageDraw

    cfg = normalize_shape(cfg)
    size = int(size)
    scale = 2
    S = size * scale
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0) if not background
                    else (*_rgb(background), 255))
    draw = ImageDraw.Draw(img)
    prims = scene(cfg, state=state, size=S, xf=xf, phase=phase)
    # Holes (the ring's interior) are punched after the fill, before
    # the outline: we collect them and stamp a transparent (or bg) oval.
    def punch(kind, coords):
        hole_color = background or (0, 0, 0, 0)
        if isinstance(hole_color, str):
            hole_color = (*_rgb(hole_color), 255)
        stamp = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        sd = ImageDraw.Draw(stamp)
        if kind == "oval":
            sd.ellipse(_intbox(coords), fill=(255, 255, 255, 255))
        else:
            sd.polygon(_intpts(coords), fill=(255, 255, 255, 255))
        img.paste(Image.new("RGBA", (S, S), hole_color), (0, 0), stamp)

    for kind, coords, fill, outline, width in prims:
        if fill == "hole":
            punch(kind, coords)
            continue
        if kind == "oval" and fill:
            draw.ellipse(_intbox(coords), fill=fill)
        elif kind == "poly" and fill:
            draw.polygon(_intpts(coords), fill=fill)
        if kind == "oval" and outline:
            draw.ellipse(_intbox(coords), outline=outline,
                         width=max(1, int(round(width))))
        elif kind == "poly" and outline:
            pts = _intpts(coords)
            if pts[0] != pts[-1]:
                pts = pts + [pts[0]]
            draw.line(pts, fill=outline, width=max(1, int(round(width))))
        elif kind == "line":
            draw.line(_intpts(coords), fill=outline or fill or INK,
                      width=max(1, int(round(width))))
    return img.resize((size, size), Image.Resampling.LANCZOS)


def _rgb(c):
    c = parse_color(c, "#000000")
    return int(c[1:3], 16), int(c[3:5], 16), int(c[5:7], 16)


def _intpts(coords):
    return [(int(round(x)), int(round(y))) for x, y in coords]


def _intbox(box):
    x0, y0, x1, y1 = box
    return (int(round(min(x0, x1))), int(round(min(y0, y1))),
            int(round(max(x0, x1))), int(round(max(y0, y1))))


# ---------------------------------------------------------- canvas

def paint_canvas(canvas, cfg, state="idle", size=None, xf=None, phase=0.0,
                 hole_color=None):
    """Draw the scene onto a Tk Canvas (items tagged 'shape'). The
    Canvas is assumed to be `size` by `size`. hole_color is the
    background used to punch the ring (the canvas bg)."""
    canvas.delete("shape")
    cfg = normalize_shape(cfg)
    size = int(size or cfg["size"])
    hole_color = hole_color or canvas.cget("bg")
    for kind, coords, fill, outline, width in scene(
            cfg, state=state, size=size, xf=xf, phase=phase):
        fill = hole_color if fill == "hole" else (fill or "")
        outline = outline or ""
        kw = {"tags": "shape"}
        if fill:
            kw["fill"] = fill
        if outline:
            kw["outline"] = outline
            kw["width"] = width
        else:
            kw["outline"] = ""
        if kind == "oval":
            canvas.create_oval(*coords, **kw)
        elif kind == "poly":
            flat = [c for p in coords for c in p]
            canvas.create_polygon(*flat, **kw)
        elif kind == "line":
            flat = [c for p in coords for c in p]
            canvas.create_line(*flat, fill=outline or INK,
                               width=max(1.0, width), capstyle="round",
                               joinstyle="round", smooth=True,
                               tags="shape")
