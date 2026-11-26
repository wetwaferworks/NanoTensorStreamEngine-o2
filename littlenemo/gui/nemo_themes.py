#!/usr/bin/env python3
"""nemo_themes.py - the chat window's theming engine (no Tk).

CloudAtlas's official pack (gui/theme-pack/theme.json, five themes,
default Pixel Nemo) is loaded directly in that format. Each pack
theme is mapped to the GUI's palette / fonts / spacing, and its
asset paths (window icon, thinking strip, status dots, buttons,
banner, tile) stay on the theme so the window can use them.

A missing pack falls back to the built-in Night theme (purple
#A970FF and orange #FF9B42 on dark). Drop-in JSON files in the older
single-theme format, or a themes/ folder beside the settings file,
are still discovered. Invalid colours are skipped so a typo cannot
crash the UI.

See THEMES.md for both formats.
"""

import copy
import json
import os

from nemo_shapes import parse_color, PURPLE, ORANGE, shade

# ------------------------------------------------------------------ pack

def bundled_pack_dir():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "theme-pack")


def bundled_themes_dir():
    """Older single-theme JSON drop-ins (Night / Ember fallbacks)."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "themes")


def find_pack_root(search_dirs=None):
    """The folder that contains theme.json, or None."""
    cands = [bundled_pack_dir()]
    for d in search_dirs or []:
        if not d:
            continue
        cands.append(d)
        cands.append(os.path.join(d, "theme-pack"))
    seen = set()
    for folder in cands:
        folder = os.path.abspath(folder)
        if folder in seen:
            continue
        seen.add(folder)
        if os.path.isfile(os.path.join(folder, "theme.json")):
            return folder
    return None


def load_pack(search_dirs=None):
    """The official theme pack as a dict, or None if it isn't present.
    Paths in the result are left relative to pack['root']."""
    root = find_pack_root(search_dirs)
    if not root:
        return None
    path = os.path.join(root, "theme.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or not isinstance(raw.get("themes"), list):
        return None
    raw["root"] = root
    raw["path"] = path
    return raw


# Pack colour keys -> GUI palette keys. Pack uses a short set
# (bg, panel, panel2, line, text, muted, p, p2, o); the GUI has a
# slightly larger role list so every widget has a colour.
def _palette_from_pack(colors, status_colors=None):
    c = colors or {}
    st = status_colors or {}
    bg = parse_color(c.get("bg"), "#0E0E10")
    panel = parse_color(c.get("panel"), "#151517")
    panel2 = parse_color(c.get("panel2"), "#1D1D21")
    line = parse_color(c.get("line"), "#2E2E34")
    text = parse_color(c.get("text"), "#ECECF1")
    muted = parse_color(c.get("muted"), "#8E8E98")
    p = parse_color(c.get("p"), PURPLE)
    p2 = parse_color(c.get("p2"), "#D9BFFF")
    o = parse_color(c.get("o"), ORANGE)
    ok = parse_color(st.get("ready"), "#4ADE80")
    warn = parse_color(st.get("generating") or c.get("o"), o)
    err = parse_color(st.get("error"), "#FF5C6C")
    return {
        "bg": bg,
        "sidebar": panel,
        "panel": panel,
        "surface": panel2,
        "raised": shade(panel2, 0.10),
        "border": line,
        "select": shade(panel2, 0.16),
        "fg": text,
        "fg2": muted,
        "dim": muted,
        "faint": shade(muted, -0.25),
        "accent": p,
        "accent2": o,
        "think": p2,
        "think_hot": shade(p2, 0.20),
        "ok": ok,
        "warn": warn,
        "warn_bg": shade(o, -0.72),
        "error": err,
        "user_bubble": panel2,
        "assistant_fg": text,
        "icon_fill": p,
        "icon_outline": o,
        # raw pack keys, so a widget can ask for them by pack name too
        "p": p, "p2": p2, "o": o, "line": line, "text": text,
        "muted": muted, "panel2": panel2,
    }


# Font nicknames used in the pack -> families to try (first one that
# is installed wins; we never download a font).
FONT_ALIASES = {
    "vt": ["VT323", "Press Start 2P", "Fixedsys", "Consolas",
           "Cascadia Mono", "Courier New"],
    "press": ["Press Start 2P", "VT323", "Fixedsys", "Consolas"],
    "chakra": ["Chakra Petch", "Segoe UI", "Verdana", "Tahoma"],
    "inter": ["Inter", "Segoe UI", "Helvetica Neue", "Arial"],
    "space": ["Space Grotesk", "Space Mono", "Segoe UI", "Arial"],
    "jb": ["JetBrains Mono", "Cascadia Mono", "Consolas", "Courier New"],
    "exo2": ["Exo 2", "Exo", "Segoe UI", "Trebuchet MS", "Arial"],
}


NIGHT = {
    "id": "night",
    "name": "Night",
    "author": "NTSE",
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
        "accent": PURPLE,
        "accent2": ORANGE,
        "think": "#C4A6FF",
        "think_hot": "#E8DEFF",
        "ok": "#4ADE80",
        "warn": "#F5B041",
        "warn_bg": "#2B2210",
        "error": "#FF7A7A",
        "user_bubble": "#2A2438",
        "assistant_fg": "#ECECF1",
        "icon_fill": PURPLE,
        "icon_outline": ORANGE,
    },
    "fonts": {
        "family": "Segoe UI",
        "mono": "Cascadia Mono",
        "body": 11,
        "small": 9,
        "title": 13,
        "hero": 20,
        "ui_alias": None,
        "heading_alias": None,
    },
    "spacing": {
        "sidebar_width": 236,
        "stats_width": 372,
        "pad": 16,
        "composer_pad": 40,
        "icon": 32,
        "icon_lg": 64,
        "radius": 8,
    },
    "icons": "default",
}

EMBER = {
    "id": "ember",
    "name": "Ember",
    "author": "NTSE",
    "version": 1,
    "palette": {
        "bg": "#140E0C",
        "sidebar": "#1B1410",
        "panel": "#1A1210",
        "surface": "#241A16",
        "raised": "#2E221C",
        "border": "#3A2C24",
        "select": "#3A2418",
        "fg": "#F6EDE4",
        "fg2": "#E2D0C0",
        "dim": "#A89080",
        "faint": "#6E584C",
        "accent": ORANGE,
        "accent2": PURPLE,
        "think": "#FFD0A8",
        "think_hot": "#FFE8D0",
        "ok": "#86D98A",
        "warn": "#F5B041",
        "warn_bg": "#2B1C10",
        "error": "#FF8A7A",
        "user_bubble": "#3A2418",
        "assistant_fg": "#F6EDE4",
        "icon_fill": ORANGE,
        "icon_outline": PURPLE,
    },
    "fonts": dict(NIGHT["fonts"]),
    "spacing": dict(NIGHT["spacing"]),
    "icons": "default",
}

BUILTINS = {"night": NIGHT, "ember": EMBER}

# Overridden once a pack is found: Pixel Nemo is the official default.
DEFAULT_THEME_ID = "pixel"

_INT_FONT_KEYS = {"body", "small", "title", "hero"}
_INT_SPACE_KEYS = {"sidebar_width", "stats_width", "pad", "composer_pad",
                   "icon", "icon_lg", "radius"}


# ---------------------------------------------------------- conversion

def theme_from_pack_entry(entry, pack):
    """One theme.json entry -> the GUI's theme dict, with asset paths
    resolved against the pack root."""
    theme = copy.deepcopy(NIGHT)
    ident = (entry.get("id") or "").strip() or "theme"
    theme["id"] = ident
    theme["name"] = (entry.get("name") or ident).strip()
    theme["subtitle"] = (entry.get("subtitle") or "").strip()
    theme["description"] = (entry.get("description") or "").strip()
    theme["author"] = "CloudAtlas"
    theme["version"] = pack.get("version") or 1
    theme["default"] = bool(entry.get("default"))
    theme["order"] = entry.get("order")
    theme["origin"] = pack.get("path") or "pack"
    theme["pack_root"] = pack["root"]
    theme["pack_id"] = ident
    theme["folder"] = entry.get("folder") or ""
    theme["notes"] = []
    theme["palette"] = _palette_from_pack(entry.get("colors") or {},
                                          pack.get("status_colors") or {})
    theme["status_colors"] = {
        k: parse_color(v, "#888888")
        for k, v in (pack.get("status_colors") or {}).items()
    }
    fonts = entry.get("fonts") or {}
    theme["fonts"]["ui_alias"] = fonts.get("ui")
    theme["fonts"]["heading_alias"] = fonts.get("heading")
    # Prefer the pack's named family when it is installed; _fonts() in
    # the UI also consults FONT_ALIASES against the live family list.
    aliases = FONT_ALIASES.get(fonts.get("ui") or "", [])
    if aliases:
        theme["fonts"]["family"] = aliases[0]
    aliases_h = FONT_ALIASES.get(fonts.get("heading") or "", [])
    if aliases_h:
        theme["fonts"]["heading"] = aliases_h[0]
    if fonts.get("ui") in ("vt", "press", "jb"):
        theme["fonts"]["mono"] = theme["fonts"]["family"]
    radius = entry.get("corner_radius")
    try:
        theme["spacing"]["radius"] = max(0, int(radius))
    except (TypeError, ValueError):
        pass
    theme["corner_radius"] = theme["spacing"]["radius"]
    theme["status_dot_shape"] = entry.get("status_dot_shape") or "circle"
    theme["assets"] = entry.get("assets") or {}
    theme["icons"] = f"theme-strip:{ident}"
    theme["scale"] = "nearest" if ident == "pixel" or theme["corner_radius"] == 0 \
        else "bilinear"
    return theme


def _merge(raw):
    """Older single-theme JSON (or a partial dict) -> a complete theme."""
    theme = copy.deepcopy(NIGHT)
    if not isinstance(raw, dict):
        theme["id"] = "invalid"
        theme["name"] = "Invalid theme"
        return theme, ["theme file is not an object"]
    notes = []
    for key in ("id", "name", "author", "subtitle", "description"):
        val = raw.get(key)
        if isinstance(val, str) and val.strip():
            theme[key] = val.strip()
    try:
        if "version" in raw:
            theme["version"] = int(raw["version"])
    except (TypeError, ValueError):
        notes.append("version is not an integer; kept 1")
    pal = raw.get("palette") or {}
    if pal and not isinstance(pal, dict):
        notes.append("palette is not an object; kept Night colours")
        pal = {}
    # A drop-in may use pack colour keys (bg/panel/p/o) as well.
    if any(k in pal for k in ("p", "o", "text", "muted", "panel2")) \
            and "accent" not in pal:
        mapped = _palette_from_pack(pal)
        theme["palette"].update(mapped)
        pal = {k: v for k, v in pal.items() if k not in mapped}
    for k, v in pal.items():
        c = parse_color(v, None)
        if c is None:
            notes.append(f"palette.{k} is not a colour; kept the default")
            continue
        theme["palette"][k] = c
    fonts = raw.get("fonts") or {}
    if fonts and not isinstance(fonts, dict):
        notes.append("fonts is not an object; kept Night fonts")
        fonts = {}
    for k, v in fonts.items():
        if k in ("family", "mono", "heading", "ui_alias", "heading_alias"):
            if isinstance(v, str) and v.strip():
                theme["fonts"][k] = v.strip()
            else:
                notes.append(f"fonts.{k} is empty; kept the default")
        elif k in _INT_FONT_KEYS:
            try:
                n = int(v)
            except (TypeError, ValueError):
                notes.append(f"fonts.{k} is not an integer; kept the default")
                continue
            if 6 <= n <= 48:
                theme["fonts"][k] = n
            else:
                notes.append(f"fonts.{k} is out of range 6–48; kept the default")
    spacing = raw.get("spacing") or {}
    if spacing and not isinstance(spacing, dict):
        notes.append("spacing is not an object; kept Night spacing")
        spacing = {}
    if "corner_radius" in raw and "radius" not in spacing:
        spacing = dict(spacing)
        spacing["radius"] = raw["corner_radius"]
    for k, v in spacing.items():
        try:
            n = int(v)
        except (TypeError, ValueError):
            notes.append(f"spacing.{k} is not an integer; kept the default")
            continue
        if 0 <= n <= 800:
            theme["spacing"][k] = n
        else:
            notes.append(f"spacing.{k} is out of range; kept the default")
    if "radius" in theme["spacing"]:
        theme["corner_radius"] = theme["spacing"]["radius"]
    icons = raw.get("icons")
    if icons is not None:
        if isinstance(icons, str) and icons.strip():
            theme["icons"] = icons.strip()
        else:
            notes.append("icons must be a pack id or path; kept 'default'")
    if raw.get("assets"):
        theme["assets"] = raw["assets"]
    if raw.get("pack_root"):
        theme["pack_root"] = raw["pack_root"]
    return theme, notes


def theme_from_dict(raw, origin="memory"):
    theme, notes = _merge(raw)
    theme["origin"] = origin
    theme["notes"] = notes
    return theme


def load_theme_file(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return theme_from_dict({"id": "missing", "name": "Missing"},
                               origin=path)
    except json.JSONDecodeError as exc:
        t = theme_from_dict({"id": "invalid", "name": "Invalid theme"},
                            origin=path)
        t["notes"] = [f"not valid JSON: {exc}"]
        return t
    # A pack-format file dropped in as a single theme.json is handled
    # by load_pack, not here. A single-theme object is merged.
    if isinstance(raw, dict) and "themes" in raw and "palette" not in raw:
        t = theme_from_dict({"id": "pack", "name": "Theme pack"}, origin=path)
        t["notes"] = ["this is a theme pack; place it in theme-pack/"]
        return t
    stem = os.path.splitext(os.path.basename(path))[0]
    if not (isinstance(raw, dict) and raw.get("id")):
        raw = dict(raw or {})
        raw["id"] = stem
    return theme_from_dict(raw, origin=os.path.abspath(path))


def write_theme_file(theme, path):
    payload = {k: theme[k] for k in
               ("id", "name", "author", "version", "palette", "fonts",
                "spacing", "icons") if k in theme}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def write_bundled_themes(dest_dir=None):
    dest_dir = dest_dir or bundled_themes_dir()
    os.makedirs(dest_dir, exist_ok=True)
    paths = []
    for theme in (NIGHT, EMBER):
        path = os.path.join(dest_dir, f"{theme['id']}.json")
        write_theme_file(theme, path)
        paths.append(path)
    return paths


def pack_themes(search_dirs=None):
    """The official five, already converted, Pixel Nemo first, then
    the rest in pack order. Empty if the pack is not on disk."""
    pack = load_pack(search_dirs)
    if not pack:
        return []
    out = [theme_from_pack_entry(e, pack) for e in pack["themes"]
           if isinstance(e, dict) and e.get("id")]
    out.sort(key=lambda t: (0 if t.get("default") else 1, t.get("order") or 99))
    return out


def default_theme_id(search_dirs=None):
    pack = load_pack(search_dirs)
    if pack and pack.get("default_theme"):
        return str(pack["default_theme"])
    for t in pack_themes(search_dirs):
        if t.get("default"):
            return t["id"]
    return "pixel" if pack else "night"


def discover_themes(search_dirs=None):
    """Official pack themes first (when present), then any drop-in
    JSON, then Night/Ember only if the pack is missing so a bare
    checkout still has something to pick."""
    found = {}
    official = pack_themes(search_dirs)
    for theme in official:
        found[theme["id"]] = theme
    # Older single-theme drop-ins (do not override a pack id).
    # When the official pack is present, skip the bundled Night/Ember
    # files so the picker shows the five approved themes; a themes/
    # folder beside the settings file can still add more.
    dirs = []
    if not official:
        dirs.append(bundled_themes_dir())
    for d in search_dirs or []:
        if d and d not in dirs:
            dirs.append(d)
            dirs.append(os.path.join(d, "themes"))
    for folder in dirs:
        if not folder or not os.path.isdir(folder):
            continue
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            if not name.lower().endswith(".json") or name == "theme.json":
                continue
            path = os.path.join(folder, name)
            theme = load_theme_file(path)
            if "not valid JSON" in ";".join(theme.get("notes") or []):
                continue
            if theme["id"] in found:
                continue
            found[theme["id"]] = theme
    if not official:
        for ident, theme in BUILTINS.items():
            found.setdefault(ident, theme_from_dict(theme, origin="builtin"))
        head = [found[k] for k in ("night", "ember") if k in found]
        rest = sorted((t for k, t in found.items() if k not in ("night", "ember")),
                      key=lambda t: t.get("name") or t["id"])
        return head + rest
    # Pack present: official order, then extra drop-ins.
    rest = [t for k, t in found.items()
            if k not in {x["id"] for x in official}]
    rest.sort(key=lambda t: t.get("name") or t["id"])
    return official + rest


def get_theme(theme_id, search_dirs=None):
    """The named theme, or the pack default (Pixel Nemo) / Night."""
    wanted = (theme_id or "").strip() or default_theme_id(search_dirs)
    for theme in discover_themes(search_dirs):
        if theme["id"] == wanted:
            return theme
    # Fallback: official default, then Night.
    fallback = default_theme_id(search_dirs)
    for theme in discover_themes(search_dirs):
        if theme["id"] == fallback:
            t = copy.deepcopy(theme)
            if wanted != fallback:
                t["notes"] = list(t.get("notes") or []) + [
                    f"theme {wanted!r} was not found; using {fallback}"]
            return t
    night = theme_from_dict(NIGHT, origin="builtin")
    if wanted != "night":
        night["notes"] = [f"theme {wanted!r} was not found; using Night"]
    return night


# ---------------------------------------------------------- assets

def _resolve(theme, rel):
    if not rel:
        return None
    if os.path.isabs(rel) and os.path.exists(rel):
        return rel
    root = theme.get("pack_root") or ""
    path = os.path.join(root, rel.replace("/", os.sep))
    return path if os.path.exists(path) else None


def window_icon_path(theme, size=32):
    """A PNG path for the window icon, closest to `size`."""
    assets = theme.get("assets") or {}
    pngs = assets.get("icon_png") or {}
    # keys may be str or int
    available = {}
    for k, rel in pngs.items():
        try:
            available[int(k)] = rel
        except (TypeError, ValueError):
            continue
    if not available:
        # fall back to the .ico only as a last resort (Windows)
        return _resolve(theme, assets.get("ico"))
    best = min(available, key=lambda s: (abs(s - int(size)), -s))
    return _resolve(theme, available[best])


def window_ico_path(theme):
    return _resolve(theme, (theme.get("assets") or {}).get("ico"))


def banner_path(theme):
    return _resolve(theme, (theme.get("assets") or {}).get("banner"))


def tile_path(theme):
    return _resolve(theme, (theme.get("assets") or {}).get("background_tile"))


def tray_icon_path(theme, size=16):
    trays = (theme.get("assets") or {}).get("tray") or {}
    rel = trays.get(str(int(size))) or trays.get(size) or trays.get("16")
    return _resolve(theme, rel)


def button_path(theme, action, size=24, style="glyph"):
    """action is one of send, stop, attach-image, attach-video,
    attach-file, new-chat, settings, model-switch, copy, clear.
    style is 'glyph' (for dark surfaces) or 'plated'."""
    buttons = (theme.get("assets") or {}).get("buttons") or {}
    entry = buttons.get(action) or {}
    styles = entry.get(style) or entry.get("glyph") or {}
    rel = styles.get(str(int(size))) or styles.get(size)
    if rel:
        path = _resolve(theme, rel)
        if path:
            return path
    # closest size
    for cand in ("24", "48", 24, 48):
        rel = styles.get(cand) if isinstance(styles, dict) else None
        if rel:
            path = _resolve(theme, rel)
            if path:
                return path
    return None


# Protocol / GUI states -> pack status names
STATUS_FOR_STATE = {
    "idle": "ready",
    "done": "ready",
    "thinking": "thinking",
    "streaming": "generating",
    "expanding": "expanding",
    "dtc": "dense compute",
    "error": "error",
}


def status_dot_path(theme, state, size=16):
    assets = (theme.get("assets") or {}).get("status") or {}
    name = STATUS_FOR_STATE.get(state, state)
    # theme.json uses "dense compute"; files use dense-compute-
    entry = assets.get(name) or assets.get(name.replace(" ", "-")) or {}
    rel = entry.get(str(int(size))) or entry.get(size)
    path = _resolve(theme, rel) if rel else None
    if path:
        return path
    for cand in ("16", "32", 16, 32):
        rel = entry.get(cand) if isinstance(entry, dict) else None
        if rel:
            path = _resolve(theme, rel)
            if path:
                return path
    return None


def thinking_strip_info(theme):
    info = (theme.get("assets") or {}).get("thinking_strip") or {}
    path = _resolve(theme, info.get("path"))
    if not path:
        return None
    return {
        "path": path,
        "frame_w": int(info.get("frame_w") or 16),
        "frame_h": int(info.get("frame_h") or 16),
        "frames": int(info.get("frames") or 12),
        "ms": int(info.get("suggested_ms_per_frame") or 80),
    }


def _thinking_frames(frames16, scale):
    frames = {16: frames16}
    for sz in (32, 64):
        bank = scale(frames16, sz)
        if bank:
            frames[sz] = bank
    return frames


def thinking_icon_pack(theme):
    """The theme's 16x16x12 thinking strip as an IconPack, scaled to
    32 and 64 (nearest-neighbour for Pixel Nemo so pixels stay square).
    Returns None if the strip is missing."""
    import nemo_icons as ic
    from PIL import Image

    info = thinking_strip_info(theme)
    if not info:
        return None
    im = Image.open(info["path"]).convert("RGBA")
    w, h = im.size
    fw, fh = info["frame_w"], info["frame_h"]
    n = info["frames"]
    if w < fw * n or h < fh:
        # tolerate a strip whose frame size is inferred
        fw = h
        n = w // fw if fw else 0
    if n <= 0 or fw <= 0:
        return None
    frames16 = [im.crop((i * fw, 0, (i + 1) * fw, fh)) for i in range(n)]
    import nemo_scale as _sc
    filt = (_sc.NEAREST if theme.get("scale") == "nearest"
            else _sc.LANCZOS3)

    def scale(bank, size):
        out = []
        for f in bank:
            if f.size == (size, size):
                out.append(f)
                continue
            scaled = _sc.scale_image(f, size, size, filt)
            if scaled is None:
                return None
            out.append(scaled)
        return out

    fps = 1000.0 / max(1, info["ms"])
    # A looping strip: motion states play the whole run; idle / done
    # hold the first frame; error holds the last.
    last = n - 1
    states = {
        "idle": {"frames": [0], "fps": 1, "loop": True},
        "thinking": {"frames": list(range(n)), "fps": fps, "loop": True},
        "streaming": {"frames": list(range(n)), "fps": fps, "loop": True},
        "expanding": {"frames": list(range(n)), "fps": fps, "loop": True},
        "dtc": {"frames": list(range(n)), "fps": max(6, fps), "loop": True,
                "motion": "pulse"},
        "error": {"frames": [last], "fps": 1, "loop": False},
        "done": {"frames": [0], "fps": 1, "loop": False},
    }
    return ic.IconPack(
        ident=f"theme-strip:{theme['id']}",
        name=f"{theme.get('name') or theme['id']} thinking",
        source=info["path"],
        frames=_thinking_frames(frames16, scale),
        states=states,
        fps=fps,
    )


def list_thinking_packs(search_dirs=None):
    """Official theme thinking strips, pickable as theme-strip:<id>."""
    out = []
    for theme in pack_themes(search_dirs):
        pack = thinking_icon_pack(theme)
        if pack is None:
            continue
        out.append({"id": pack.id, "name": pack.name, "source": pack.source,
                    "fallback": False})
    return out
