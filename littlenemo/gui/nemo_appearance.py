#!/usr/bin/env python3
"""nemo_appearance.py - theme, icon and shape settings in the INI (no Tk).

Appearance is stored beside the engine settings, in the same
littlenemo.ini, under [appearance] and [model.<key>]. Engine
resolve_settings / apply_settings are left alone so their tests stay
byte-identical. Saves go through update_ini_text, so comments and
unrelated keys survive.

A user-defined model is a [model.<key>] section whose key is not
SuperNemo or Little Nemo; it shows up in the sidebar with its own
icon and, if engine=/host= are set, can be talked to.
"""

import configparser
import os

import nemo_chat as proto
import nemo_icons as icons
import nemo_shapes as shapes
import nemo_themes as themes

APPEARANCE_KEYS = ("theme",)
MODEL_APPEAR_KEYS = ("icon", "shape", "fill", "outline", "face",
                     "size", "motion", "speed")
# Optional keys a user model may carry (in addition to appearance).
USER_MODEL_KEYS = ("label", "full", "engine", "host", "port", "inputs")


def _ini_dir(ini_path):
    return os.path.dirname(os.path.abspath(ini_path)) if ini_path else os.getcwd()


def theme_search_dirs(ini_path=None):
    dirs = [themes.bundled_themes_dir()]
    if ini_path:
        dirs.append(os.path.join(_ini_dir(ini_path), "themes"))
    return dirs


def icon_search_dirs(ini_path=None):
    dirs = [icons.bundled_icons_dir()]
    if ini_path:
        dirs.append(os.path.join(_ini_dir(ini_path), "icons"))
    return dirs


def default_appearance():
    models = {}
    for key in proto.MODEL_ORDER:
        models[key] = dict(shapes.DEFAULT_SHAPES.get(key)
                           or shapes.DEFAULT_SHAPES["default"])
    return {"theme": themes.default_theme_id(), "models": models}


def _read_cfg(ini_path):
    cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
    if ini_path and os.path.isfile(ini_path):
        cfg.read(ini_path, encoding="utf-8")
    return cfg


def _section_models(cfg):
    """[model.foo] section names -> keys."""
    out = []
    for sec in cfg.sections():
        low = sec.lower()
        if low.startswith("model.") and len(low) > 6:
            out.append((low[6:], sec))
    return out


def load_appearance(ini_path):
    """Read theme + per-model icon/shape settings. Missing file or
    keys yield the built-in defaults. Values are normalized (bad
    colours, motions, sizes fall back)."""
    data = default_appearance()
    cfg = _read_cfg(ini_path)
    if cfg.has_section("appearance"):
        tid = cfg.get("appearance", "theme", fallback="").strip()
        if tid:
            data["theme"] = tid
    # Built-ins first, then any extra [model.*] sections.
    seen = set()
    for key, sec in _section_models(cfg):
        raw = {k: cfg.get(sec, k, fallback="").strip()
               for k in list(MODEL_APPEAR_KEYS) + list(USER_MODEL_KEYS)
               if cfg.has_option(sec, k)}
        # face may be a bool-ish string; normalize_shape handles it.
        cfg_shape = {k: raw[k] for k in MODEL_APPEAR_KEYS if k in raw}
        ident = key if key in shapes.DEFAULT_SHAPES else "default"
        model = shapes.normalize_shape(cfg_shape, ident)
        for k in USER_MODEL_KEYS:
            if k in raw:
                model[k] = raw[k]
        data["models"][key] = model
        seen.add(key)
    return data


def appearance_updates(data):
    """(section, key, value) triples for update_ini_text."""
    updates = [("appearance", "theme",
                (data.get("theme") or themes.DEFAULT_THEME_ID).strip()
                or themes.DEFAULT_THEME_ID)]
    for key, cfg in (data.get("models") or {}).items():
        ident = key if key in shapes.DEFAULT_SHAPES else "default"
        n = shapes.normalize_shape(cfg, ident)
        sec = f"model.{key}"
        updates.append((sec, "icon", n["icon"]))
        updates.append((sec, "shape", n["shape"]))
        updates.append((sec, "fill", n["fill"]))
        updates.append((sec, "outline", n["outline"]))
        updates.append((sec, "face", "true" if n["face"] else "false"))
        updates.append((sec, "size", str(n["size"])))
        updates.append((sec, "motion", n["motion"]))
        updates.append((sec, "speed",
                        f"{n['speed']:.2f}".rstrip("0").rstrip(".")))
        for k in USER_MODEL_KEYS:
            if cfg.get(k):
                updates.append((sec, k, str(cfg[k]).strip()))
    return updates


def save_appearance(ini_path, data):
    """Write appearance into the INI, preserving comments and
    unrelated keys. Creates the file if needed. Returns the path."""
    if not ini_path:
        raise ValueError("no settings file to write to")
    text = ""
    if os.path.isfile(ini_path):
        with open(ini_path, "r", encoding="utf-8") as f:
            text = f.read()
    os.makedirs(os.path.dirname(os.path.abspath(ini_path)) or ".",
                exist_ok=True)
    tmp = ini_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(proto.update_ini_text(text, appearance_updates(data)))
    os.replace(tmp, ini_path)
    return ini_path


def model_cfg(data, key):
    models = (data or {}).get("models") or {}
    if key in models:
        ident = key if key in shapes.DEFAULT_SHAPES else "default"
        return shapes.normalize_shape(models[key], ident)
    return shapes.normalize_shape(None, key)


def set_model_cfg(data, key, cfg):
    data = dict(data or default_appearance())
    models = dict(data.get("models") or {})
    ident = key if key in shapes.DEFAULT_SHAPES else "default"
    prev = dict(models.get(key) or {})
    prev.update(cfg or {})
    models[key] = shapes.normalize_shape(prev, ident)
    # Keep user-model metadata.
    for k in USER_MODEL_KEYS:
        if k in prev and prev[k]:
            models[key][k] = prev[k]
    data["models"] = models
    return data


def catalog(ini_path, data=None):
    """Models shown in the sidebar: built-ins in MODEL_ORDER, then
    any extra [model.*] sections (user models)."""
    data = data or load_appearance(ini_path)
    items = []
    seen = set()
    for key in proto.MODEL_ORDER:
        info = dict(proto.MODELS[key])
        info["key"] = key
        info["builtin"] = True
        info["appearance"] = model_cfg(data, key)
        items.append(info)
        seen.add(key)
    for key, cfg in (data.get("models") or {}).items():
        if key in seen:
            continue
        label = (cfg.get("label") or key).strip() or key
        items.append({
            "key": key,
            "label": label,
            "full": (cfg.get("full") or "User model").strip(),
            "inputs": tuple(
                p.strip() for p in (cfg.get("inputs") or "text").split(",")
                if p.strip() in ("text", "image", "video")
            ) or ("text",),
            "unavailable": (f"No engine is configured for {label}. "
                            f"Set engine= (or host= and port=) in "
                            f"[model.{key}] of the settings file."),
            "builtin": False,
            "appearance": model_cfg(data, key),
        })
        seen.add(key)
    return items


def extend_targets(targets, ini_path, data=None):
    """Add engine targets for user models listed in the INI."""
    targets = dict(targets or {})
    data = data or load_appearance(ini_path)
    base = _ini_dir(ini_path)
    for key, cfg in (data.get("models") or {}).items():
        if key in targets:
            continue
        engine = (cfg.get("engine") or "").strip()
        host = (cfg.get("host") or "").strip() or None
        port = (cfg.get("port") or "").strip() or None
        if engine and not os.path.isabs(engine):
            engine = os.path.normpath(os.path.join(base, engine))
        targets[key] = {"cmd": engine or None, "host": host if port else None,
                        "port": port if host else None}
    return targets


def resolve_icon_for(data, key, ini_path=None):
    """The pack and whether the GUI should draw the procedural shape
    instead (pack missing/bad/unset)."""
    cfg = model_cfg(data, key)
    pack, procedural = icons.resolve_icon(
        cfg.get("icon"), model_key=key,
        search_dirs=icon_search_dirs(ini_path))
    return cfg, pack, procedural
