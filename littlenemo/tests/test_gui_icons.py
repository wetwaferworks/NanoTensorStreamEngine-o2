#!/usr/bin/env python3
"""test_gui_icons.py - headless tests of the icon registry, fallback,
animation state machine, procedural shapes, theme loader, and
appearance persistence (the picker saving to the INI)."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "gui"))

import nemo_appearance as appearance  # noqa: E402
import nemo_icons as icons  # noqa: E402
import nemo_shapes as shapes  # noqa: E402
import nemo_chat as proto  # noqa: E402
import nemo_themes as themes  # noqa: E402
from PIL import Image  # noqa: E402

fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def main():
    # --- shapes ---
    for name in shapes.SHAPES:
        prims = shapes.shape_prims(name, 16, 16, 10)
        expect(len(prims) >= 2, f"{name} produces primitives")
        kinds = {p[0] for p in prims}
        expect(kinds <= {"poly", "oval", "line"}, f"{name} uses known kinds")
    cfg = shapes.normalize_shape({"shape": "hexagon", "fill": "blue",
                                  "motion": "zoom", "size": "nope",
                                  "speed": -3, "face": "YES"}, "supernemo")
    expect(cfg["shape"] == "blob" and cfg["fill"] == shapes.PURPLE
           and cfg["motion"] == "bob" and cfg["size"] == 32
           and cfg["speed"] == 1.0 and cfg["face"] is True,
           "invalid shape fields fall back to SuperNemo personality")
    expect(shapes.parse_color("#a97") == "#AA9977"
           and shapes.parse_color("#A970FF") == "#A970FF"
           and shapes.parse_color("purple") is None,
           "colour parser accepts #RGB/#RRGGBB only")
    face_on = shapes.scene({"shape": "orb", "face": True}, "thinking", 32)
    face_off = shapes.scene({"shape": "orb", "face": False}, "thinking", 32)
    expect(len(face_on) > len(face_off), "face adds primitives")
    img = shapes.render_pil({"shape": "blob", "fill": shapes.PURPLE,
                             "outline": shapes.ORANGE, "face": True},
                            state="thinking", size=32)
    expect(img.size == (32, 32) and img.mode == "RGBA"
           and img.getextrema()[3][1] > 0, "renderer returns a 32px RGBA")
    xf = icons.motion_transform("bob", 1.57, 32)
    expect(abs(xf["dy"]) > 1, "bob moves vertically")
    xf = icons.motion_transform("flip", 0.0, 32)
    expect(abs(xf["flip_x"] - 1.0) < 1e-6, "flip starts facing forward")
    xf = icons.motion_transform("spin", 3.14159265, 32)
    expect(abs(xf["rot"] - 180) < 1, "spin at π is 180 degrees")
    expect(icons.motion_for("expanding", "spin") == "pulse"
           and icons.motion_for("expanding", "bob") == "spin"
           and icons.motion_for("thinking", "flip") == "flip",
           "expanding stays visually distinct from the thinking motion")

    # --- state machine ---
    expect(icons.next_state("idle", "turn_begin") == "thinking",
           "Send starts thinking")
    expect(icons.next_state("thinking", "tok") == "streaming",
           "TOK moves to streaming")
    expect(icons.STATES == ("idle", "thinking", "streaming", "expanding",
                            "dtc", "error", "done"),
           "seven mascot states including dtc")
    expect(icons.next_state("streaming", "expand", {"state": "begin"})
           == "expanding", "EXPAND begin is a distinct state")
    expect(icons.next_state("thinking", "dtc", {"state": "begin"})
           == "dtc", "DTC begin is its own state")
    expect(icons.next_state("dtc", "dtc", {"state": "end"})
           == "thinking", "DTC end returns to thinking")
    expect(icons.next_state("streaming", "expand", {"state": "end"})
           == "streaming", "EXPAND end returns to streaming")
    expect(icons.motion_for("dtc", "pulse") == "spin"
           and icons.motion_for("dtc", "bob") == "pulse",
           "dtc stays visually distinct from expanding")
    expect(icons.next_state("thinking", "error") == "error", "ERROR")
    expect(icons.next_state("streaming", "done") == "done", "DONE")
    expect(icons.next_state("error", "done") == "error",
           "a failed turn stays on error through DONE")
    expect(icons.next_state("done", "idle") == "idle", "settles to idle")

    clock = [0.0]
    anim = icons.IconAnimator({"shape": "blob", "motion": "bob", "speed": 1},
                              model_key="supernemo", clock=lambda: clock[0])
    expect(anim.state == "idle" and anim.settled, "starts idle and settled")
    expect(anim.interval_ms() is None, "idle has no timer")
    anim.on_event("turn_begin", now=0.0)
    expect(anim.state == "thinking" and not anim.settled
           and anim.interval_ms() is not None, "thinking runs a timer")
    clock[0] = 0.25
    s1, f1, xf1, _ = anim.step(0.25)
    clock[0] = 0.50
    s2, f2, xf2, _ = anim.step(0.50)
    expect(s1 == s2 == "thinking" and (f1 != f2 or xf1 != xf2),
           "thinking advances a frame or a transform")
    anim.on_event("expand", {"state": "begin"}, now=0.50)
    expect(anim.state == "expanding", "event reaches expanding")
    anim.on_event("dtc", {"state": "begin"}, now=0.60)
    expect(anim.state == "dtc" and not anim.settled
           and anim.interval_ms() is not None, "dtc runs a timer")
    anim.on_event("tok", now=1.0)
    expect(anim.state == "streaming", "TOK from expanding goes streaming")
    anim.on_event("done", now=1.0)
    st, _, _, settled = anim.step(1.0)
    expect(st == "done" and not settled, "done plays a flourish")
    st, _, _, settled = anim.step(1.0 + icons.SETTLE_S["done"])
    expect(st == "done" and settled and anim.interval_ms() is None,
           "done settles and stops the timer")

    # --- pack loading and fallback ---
    tmp = tempfile.mkdtemp(prefix="nemo-icons-")
    # a valid 4-frame 32px strip, no manifest
    strip = Image.new("RGBA", (128, 32), (0, 0, 0, 0))
    for i, c in enumerate([(169, 112, 255, 255), (255, 155, 66, 255),
                           (80, 80, 80, 255), (255, 255, 255, 255)]):
        Image.new("RGBA", (32, 32), c).paste  # noqa: silly
        block = Image.new("RGBA", (32, 32), c)
        strip.paste(block, (i * 32, 0))
    bare = os.path.join(tmp, "bare-strip.png")
    strip.save(bare)
    pack = icons.load_pack_from_path(bare)
    expect(len(pack.frames[32]) == 4 and pack.states["thinking"]["frames"]
           == [1, 2, 3] and not pack.fallback, "bare strip splits into frames")

    folder = os.path.join(tmp, "handed-pack")
    os.makedirs(os.path.join(folder, "frames-32"))
    os.makedirs(os.path.join(folder, "frames-64"))
    for sz in (32, 64):
        bank = []
        for i in range(3):
            im = Image.new("RGBA", (sz, sz), (i * 40, 10, 200, 255))
            im.save(os.path.join(folder, f"frames-{sz}", f"{i:03d}.png"))
            bank.append(im)
        wide = Image.new("RGBA", (sz * 3, sz), (0, 0, 0, 0))
        for i, im in enumerate(bank):
            wide.paste(im, (i * sz, 0))
        wide.save(os.path.join(folder, f"strip-{sz}.png"))
    with open(os.path.join(folder, "manifest.json"), "w") as f:
        json.dump({
            "name": "Handed", "sizes": [32, 64], "fps": 10, "format": "frames",
            "states": {
                "idle": {"frames": [0], "fps": 1, "loop": True},
                "thinking": {"frames": [1, 2], "fps": 10, "loop": True,
                             "motion": "wobble"},
            }
        }, f)
    pack = icons.load_pack_from_path(folder)
    expect(pack.name == "Handed" and pack.states["thinking"]["frames"] == [1, 2]
           and pack.states["thinking"]["motion"] == "wobble"
           and "idle" in pack.states and "done" in pack.states,
           "manifest pack loads and fills missing states from idle")

    # bad packs
    def rejects(path, needle):
        try:
            icons.load_pack_from_path(path)
            return False
        except ValueError as exc:
            return needle in str(exc)

    missing = os.path.join(tmp, "nope")
    expect(rejects(missing, "not found"), "missing path rejected")
    badj = os.path.join(tmp, "badjson")
    os.makedirs(badj)
    with open(os.path.join(badj, "manifest.json"), "w") as f:
        f.write("{not json")
    expect(rejects(badj, "JSON"), "bad manifest rejected")
    empty = os.path.join(tmp, "empty")
    os.makedirs(empty)
    with open(os.path.join(empty, "manifest.json"), "w") as f:
        json.dump({"name": "x", "states": {"thinking": {"frames": [9]}}}, f)
    expect(rejects(empty, ""), "out-of-range / missing frames rejected")

    pack, proc = icons.resolve_icon("", "supernemo")
    expect(proc and pack.fallback and "thinking shape" in pack.reason,
           "empty icon= uses the procedural shape")
    pack, proc = icons.resolve_icon(folder, "supernemo")
    expect(not proc and pack.name == "Handed", "valid path is used")
    pack, proc = icons.resolve_icon(missing, "littlenemo")
    expect(proc and pack.fallback and pack.id.startswith("builtin:"),
           "missing pack falls back to the built-in")
    pack, proc = icons.resolve_icon(badj, "supernemo")
    expect(proc and pack.fallback, "bad pack falls back")
    pack, proc = icons.resolve_icon("builtin:littlenemo", "littlenemo")
    expect(not proc and pack.id == "builtin:littlenemo",
           "builtin: id loads the in-memory pack")
    found = icons.discover_packs([tmp, icons.bundled_icons_dir()])
    ids = [p["id"] for p in found]
    expect(any(p["id"] == "builtin:supernemo" for p in found)
           and any("handed-pack" in i for i in ids)
           and any("bare-strip.png" in i for i in ids),
           "discover finds built-ins, folders and bare strips")

    # bundled placeholder packs written for CloudAtlas
    bundled = os.path.join(icons.bundled_icons_dir(), "supernemo")
    if os.path.isdir(bundled):
        pack, proc = icons.resolve_icon(bundled, "supernemo")
        expect(not proc and pack.frames[32] and pack.frames[64]
               and pack.states["thinking"]["frames"],
               "shipped SuperNemo pack loads")

    # --- themes ---
    night = themes.theme_from_dict(themes.NIGHT, origin="builtin")
    expect(night["palette"]["accent"] == "#A970FF"
           and night["palette"]["accent2"] == "#FF9B42"
           and night["id"] == "night", "Night is purple and orange")
    ember = themes.theme_from_dict(themes.EMBER, origin="builtin")
    expect(ember["palette"]["accent"] == "#FF9B42"
           and ember["id"] == "ember", "Ember is a second built-in")

    pack = themes.load_pack()
    expect(pack is not None and pack.get("default_theme") == "pixel"
           and themes.default_theme_id() == "pixel",
           "official pack is present and defaults to Pixel Nemo")
    official = themes.discover_themes()
    ids = [t["id"] for t in official]
    expect(ids[:5] == ["pixel", "lattice", "stream", "handshake", "owl"]
           and "night" not in ids and "ember" not in ids,
           "picker lists the five official themes, pixel first")
    pixel = themes.get_theme("pixel")
    expect(pixel["id"] == "pixel"
           and pixel["palette"]["accent"] == "#A970FF"
           and pixel["palette"]["accent2"] == "#FF9A3C"
           and pixel["spacing"]["radius"] == 0
           and pixel.get("scale") == "nearest",
           "Pixel Nemo maps pack colours and uses nearest-neighbour scale")
    ico = themes.window_ico_path(pixel)
    png = themes.window_icon_path(pixel, 32)
    banner = themes.banner_path(pixel)
    glyph = themes.button_path(pixel, "send", 24, "glyph")
    dot = themes.status_dot_path(pixel, "dtc", 16)
    strip = themes.thinking_strip_info(pixel)
    tpack = themes.thinking_icon_pack(pixel)
    expect(ico and os.path.isfile(ico) and ico.endswith(".ico")
           and png and os.path.isfile(png)
           and banner and os.path.isfile(banner)
           and glyph and os.path.isfile(glyph)
           and "glyph" in glyph.replace("\\", "/")
           and dot and os.path.isfile(dot)
           and "dense-compute" in dot.replace("\\", "/")
           and strip and strip["frames"] == 12 and strip["frame_w"] == 16
           and tpack is not None and tpack.id == "theme-strip:pixel"
           and "dtc" in tpack.states
           and 32 in tpack.frames and 64 in tpack.frames,
           "pack assets resolve: ico, png, banner, glyph, dtc status, thinking strip")
    missing = themes.get_theme("does-not-exist")
    expect(missing["id"] == "pixel" and any("not found" in n
                                            for n in missing["notes"]),
           "unknown theme falls back to Pixel Nemo")
    # Handshake banner exists; THEMES.md notes HandshakeOS is not public-cleared.
    hs = themes.get_theme("handshake")
    hs_banner = themes.banner_path(hs)
    expect(hs["id"] == "handshake" and hs_banner and os.path.isfile(hs_banner),
           "Handshake theme and banner are selectable")

    tdir = os.path.join(tmp, "themes")
    os.makedirs(tdir)
    with open(os.path.join(tdir, "reef.json"), "w") as f:
        json.dump({"id": "reef", "name": "Reef",
                   "palette": {"accent": "#00AABB", "nope": "red",
                               "bg": "#102030"},
                   "fonts": {"body": 99, "family": "Inter"}}, f)
    reef = themes.get_theme("reef", search_dirs=[tdir])
    expect(reef["id"] == "reef" and reef["palette"]["accent"] == "#00AABB"
           and reef["palette"]["bg"] == "#102030"
           and reef["palette"]["accent2"] == "#FF9B42"
           and reef["fonts"]["body"] == 11  # 99 out of range
           and reef["fonts"]["family"] == "Inter",
           "drop-in theme merges, rejects bad values, keeps Night fill-ins")
    expect(any("palette.nope" in n or "body" in n for n in reef["notes"]),
           "invalid theme keys are reported")
    with open(os.path.join(tdir, "broken.json"), "w") as f:
        f.write("{")
    listed = themes.discover_themes([tdir])
    ids = [t["id"] for t in listed]
    expect("pixel" in ids and "reef" in ids and "night" not in ids,
           "discover lists official pack and drop-ins, not Night")
    # A file that is not valid JSON must not replace Pixel.
    expect(themes.get_theme("pixel", search_dirs=[tdir])["id"] == "pixel"
           and themes.get_theme("pixel", search_dirs=[tdir])["palette"]["accent"]
           == "#A970FF", "a broken drop-in cannot replace Pixel Nemo")

    pack2, proc2 = icons.resolve_icon("theme-strip:pixel", "supernemo")
    expect(not proc2 and pack2.id == "theme-strip:pixel"
           and pack2.frames[32] and pack2.frames[64],
           "theme-strip:pixel resolves to the scaled thinking pack")

    # --- appearance persistence (the picker) ---
    ini = os.path.join(tmp, "littlenemo.ini")
    with open(ini, "w") as f:
        f.write("; keep me\n[paths]\nengine = ./e\n\n[notes]\ncustom_key = keepme\n")
    data = appearance.load_appearance(ini)
    expect(data["theme"] == "pixel"
           and data["models"]["supernemo"]["shape"] == "blob"
           and data["models"]["littlenemo"]["shape"] == "droplet",
           "missing appearance keys yield Pixel Nemo and built-in shapes")
    data = appearance.set_model_cfg(data, "supernemo", {
        "icon": folder, "shape": "star", "fill": "#112233",
        "outline": "#AABBCC", "face": False, "size": 64,
        "motion": "spin", "speed": 1.5,
    })
    data["theme"] = "ember"
    # a user model too
    data["models"]["my-llama"] = shapes.normalize_shape({
        "shape": "cube", "fill": "#A970FF", "icon": "",
    }, "default")
    data["models"]["my-llama"]["label"] = "My Llama"
    appearance.save_appearance(ini, data)
    with open(ini) as f:
        text = f.read()
    expect("; keep me" in text and "custom_key = keepme" in text
           and "theme = ember" in text
           and "icon =" in text and folder in text
           and "shape = star" in text and "face = false" in text
           and "motion = spin" in text
           and "[model.my-llama]" in text and "label = My Llama" in text,
           "picker save writes appearance and preserves unrelated INI")
    data2 = appearance.load_appearance(ini)
    expect(data2["theme"] == "ember"
           and data2["models"]["supernemo"]["shape"] == "star"
           and data2["models"]["supernemo"]["icon"] == folder
           and data2["models"]["supernemo"]["face"] is False
           and data2["models"]["supernemo"]["size"] == 64
           and abs(data2["models"]["supernemo"]["speed"] - 1.5) < 1e-6
           and data2["models"]["my-llama"]["label"] == "My Llama",
           "picker selection persists through a load")
    cat = appearance.catalog(ini, data2)
    keys = [m["key"] for m in cat]
    expect(keys[:2] == ["supernemo", "littlenemo"] and "my-llama" in keys
           and not [m for m in cat if m["key"] == "my-llama"][0]["builtin"],
           "catalog lists built-ins then user models")
    cfg, pack, proc = appearance.resolve_icon_for(data2, "supernemo", ini)
    expect(not proc and pack.name == "Handed",
           "persisted icon path resolves to the pack (overrides the shape)")
    cfg, pack, proc = appearance.resolve_icon_for(data2, "my-llama", ini)
    expect(proc, "user model with empty icon= uses the thinking shape")

    # --- --theme CLI (session override; does not touch the INI) ---
    expect(proto._arg(["--theme", "lattice", "--model", "supernemo"],
                      "--theme") == "lattice",
           "--theme is parsed from argv")
    expect(proto._arg(["--model", "supernemo"], "--theme") is None,
           "--theme is optional")
    for tid in ("pixel", "lattice", "stream", "handshake", "owl"):
        expect(themes.get_theme(tid)["id"] == tid,
               f"--theme {tid} names a real pack theme")

    # --- official ntse-mascot/1 packs ---
    # Partial-alpha blit: 50% red over opaque black becomes mid-red,
    # not a hard cutout and not a straight copy of the source alpha.
    bg = Image.new("RGBA", (2, 2), (0, 0, 0, 255))
    src = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    blended = icons.alpha_blit(bg, src, (0, 0))
    px = blended.getpixel((0, 0))
    expect(px[0] in (127, 128) and px[1] == 0 and px[2] == 0 and px[3] == 255,
           "alpha_blit blends partial alpha over an opaque backdrop")
    # A source that hangs off the edge still composites the overlap.
    off = icons.alpha_blit(Image.new("RGBA", (4, 4), (0, 0, 255, 255)),
                           Image.new("RGBA", (4, 4), (255, 255, 0, 128)),
                           (-1, -1))
    expect(off.size == (4, 4) and off.getpixel((0, 0))[2] in (127, 128),
           "alpha_blit clips a negatively offset source")

    official = {
        "pixel": "pixel-nemo",
        "lattice": "lattice-cube",
        "stream": "stream-critter",
        "handshake": "handshake-oO",
        "owl": "sage-owl",
    }
    for tid, ident in official.items():
        pack = icons.load_pack_from_path(
            os.path.join(icons.bundled_icons_dir(), ident))
        idle_idx, idle_fps, idle_loop = pack.state_frames("idle")
        dtc_idx, dtc_fps, dtc_loop = pack.state_frames("dtc")
        done_idx, done_fps, done_loop = pack.state_frames("done")
        expect(pack.name and pack.theme == tid and not pack.apply_motion
               and len(idle_idx) == 8 and abs(idle_fps - 6) < 1e-6
               and idle_loop is True
               and len(dtc_idx) == 12 and dtc_loop is True
               and len(done_idx) == 10 and done_loop is False
               and pack.states["idle"].get("motion") == "bob"
               and pack.states["streaming"].get("motion") == "bob"
               and pack.states["idle"].get("description")
               and pack.image("thinking", 0, 32) is not None
               and pack.image("thinking", 0, 32).size == (32, 32)
               and pack.image("dtc", 0, 64).size == (64, 64),
               f"official mascot {ident} loads (v1 keys, repeated bob)")
        themed = icons.theme_mascot_pack(tid)
        expect(themed is not None and os.path.basename(themed.id) == ident,
               f"theme_mascot_pack({tid}) finds {ident}")

    # fallback_state fills states the pack leaves out; extra keys
    # (loop, fallback_state, description) and a repeated motion tag
    # must not be rejected.
    fbdir = os.path.join(tmp, "fallback-pack")
    os.makedirs(fbdir)
    # 4-frame 32px strip (solid colours, some partial alpha on frame 1)
    strip = Image.new("RGBA", (128, 32), (0, 0, 0, 0))
    for i, color in enumerate(((160, 80, 255, 255),
                               (255, 154, 60, 128),
                               (80, 200, 120, 255),
                               (255, 80, 100, 200))):
        cell = Image.new("RGBA", (32, 32), color)
        strip.paste(cell, (i * 32, 0))
    strip.save(os.path.join(fbdir, "idle-32.png"))
    with open(os.path.join(fbdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({
            "schema": "ntse-mascot/1",
            "id": "fallback-pack",
            "name": "Fallback Pack",
            "theme": "pixel",
            "sizes": [32],
            "scaling": "nearest-integer",
            "fallback_state": "idle",
            "states": {
                "idle": {
                    "frames": 4,
                    "fps": 6,
                    "motion": "bob",
                    "loop": True,
                    "strip": {"32": "idle-32.png"},
                    "description": "a four-frame idle",
                },
                "thinking": {
                    "frames": 4,
                    "fps": 10,
                    "motion": "bob",   # repeated motion tag, same as idle
                    "loop": True,
                    "strip": {"32": "idle-32.png"},
                    "description": "also bob",
                },
            },
        }, f)
    fb = icons.load_pack_from_path(fbdir)
    expect(fb.states["idle"]["motion"] == "bob"
           and fb.states["thinking"]["motion"] == "bob"
           and fb.states["idle"].get("description") == "a four-frame idle"
           and fb.states["streaming"]["frames"] == fb.states["idle"]["frames"]
           and fb.states["error"]["frames"] == fb.states["idle"]["frames"]
           and fb.states["done"]["frames"] == fb.states["idle"]["frames"]
           and fb.states["dtc"]["frames"] == fb.states["idle"]["frames"]
           and not fb.apply_motion
           and fb.image("error", 0, 32) is not None,
           "fallback_state, loop, description and repeated motion are accepted")

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
