#!/usr/bin/env python3
"""nemo_demo.py - scripted walk-throughs of the chat window, for the
Tk smoke test (tests/test_gui_ui_smoke.py) and for screenshots.

    python nemo_chat.py --model supernemo \
        --supernemo-engine "python ../tests/fake_engine.py --stdio" \
        --engine "python ../tests/fake_engine.py --stdio --model littlenemo" \
        --demo tour --shots ./shots

Demos drive the real widgets through the same methods a click does;
the engines are the test fixture, whose replies say they come from a
fixture. Names: tour (screenshots), smoke (same checks, no shots),
unavailable (SuperNemo with no engine configured), composer (the
multi-line input bar, the About and credits dialog and the settings
dialog, with screenshots).
"""

import os
import tempfile


def make_samples(folder=None):
    """A small PNG and a 6-frame animated GIF to attach."""
    from PIL import Image, ImageDraw

    folder = folder or tempfile.mkdtemp(prefix="nemo-demo-")
    png = os.path.join(folder, "workbench-sample.png")
    im = Image.new("RGB", (480, 320))
    d = ImageDraw.Draw(im)
    for y in range(320):
        d.line([(0, y), (480, y)], fill=(20 + y // 4, 40 + y // 5, 90 + y // 3))
    d.rectangle([60, 70, 200, 250], fill=(230, 160, 60))
    d.ellipse([250, 90, 410, 250], fill=(90, 200, 160))
    d.text((70, 270), "NTSE demo image", fill=(255, 255, 255))
    im.save(png)
    gif = os.path.join(folder, "orbit-clip.gif")
    frames = []
    for i in range(6):
        f = Image.new("RGB", (240, 160), (18, 18, 28))
        fd = ImageDraw.Draw(f)
        x = 30 + i * 32
        fd.ellipse([x, 60, x + 40, 100], fill=(240, 200, 80))
        fd.text((8, 8), f"frame {i}", fill=(200, 200, 220))
        frames.append(f)
    frames[0].save(gif, save_all=True, append_images=frames[1:], duration=120,
                   loop=0)
    return png, gif


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def _chat_text(app):
    return app.chat.get("1.0", "end")


def _row(app, section, label):
    for sec, items in app.stats.rows():
        if sec == section:
            for k, v in items:
                if k == label:
                    return v
    return None


def script(app, name):
    if name in ("tour", "smoke"):
        yield from _tour(app)
    elif name == "unavailable":
        yield from _unavailable(app)
    elif name == "pickers":
        yield from _pickers(app)
    elif name == "resize":
        yield from _resize(app)
    elif name == "mascot":
        yield from _mascot(app)
    elif name == "composer":
        yield from _composer(app)
    elif name == "sizes":
        yield from _sizes(app)
    else:
        raise ValueError(f"unknown demo {name!r}")


def _tour(app):
    import nemo_chat as proto
    import nemo_stats as stats

    _check(proto.target_available(app.targets.get("supernemo")),
           "the tour needs --supernemo-engine (the fixture)")
    if app.model != "supernemo":
        app.select_model("supernemo")
    yield ("wait", lambda: app.connected, 20, "SuperNemo fixture READY")
    yield ("wait", lambda: bool(app.stats.probe), 30, "hardware probe")
    png, gif = make_samples()
    app.add_attachment(png, "image")
    app.add_attachment(gif, "video")
    yield ("wait", lambda: all(a["state"] != "checking"
                               for a in app.attachments), 20, "attachments")
    _check(all(a["state"] == "ready" for a in app.attachments),
           f"attachments not ready: {[a['note'] for a in app.attachments]}")
    _check(len(app.attachments[1]["frames"]) == 6, "GIF frames extracted")
    app.input.insert("1.0", "Describe what's in this picture and this short "
                     "clip, and tell me what you're thinking as you go.")
    app._input_changed()
    yield ("shot", "01-composer-with-attachments.png")
    app.on_send()
    _check(app.busy, "turn started")
    yield ("wait", lambda: app.cur is not None and app.cur.chars > 120
           and not app.cur.answering, 20, "live thinking")
    yield ("sleep", 0.3)
    _check(app.cur.line.cget("text") != "", "thinking line shows a headline")
    yield ("shot", "02-thinking-line-live.png")
    yield ("shot", "09-thinking-icon-animating.png")
    # Screenshots take ~1s; a fast fixture may already be streaming or
    # even finished. Treat "done" as having started the answer.
    if not ((app.cur is not None and app.cur.answering) or not app.busy):
        yield ("wait", lambda: (app.cur is not None and app.cur.answering)
               or not app.busy, 30, "answer starts")
    yield ("sleep", 0.6)
    yield ("shot", "03-answer-streaming-stats.png")
    yield ("wait", lambda: not app.busy, 60, "turn DONE")
    txt = _chat_text(app)
    _check("test fixture" in txt, "fixture reply rendered")
    _check("2 attachment(s)" in txt, "both attachments reached the engine")
    tl = app.thinklines[-1]
    _check(tl.done and tl.title.cget("text").startswith("Thought for"),
           "thinking line finished")
    tl.toggle()
    _check(tl.expanded, "thinking expands")
    ttft = _row(app, "Throughput", "Time to first token")
    _check(ttft and ttft != stats.DASH, f"TTFT shown ({ttft})")
    _check(_row(app, "Now", "In DTC") not in (None, stats.DASH), "DTC shown")
    _check(_row(app, "Tensor Handshake", "Prefetch hits / misses")
           not in (None, stats.DASH), "handshake shown")
    _check(_row(app, "Hardware", "CPU") not in (None, stats.DASH), "CPU shown")
    yield ("sleep", 0.3)
    yield ("shot", "04-reply-thinking-expanded.png")

    # STOP mid-reply
    app.input.insert("1.0", "Count slowly, then stop when I say.")
    app.on_send()
    yield ("wait", lambda: app.cur is not None and app.cur.chars > 30, 20,
           "second turn thinking")
    app.on_stop()
    yield ("wait", lambda: not app.busy, 20, "stopped turn DONE")
    _check(bool(app.last_stats.get("stopped")), "STOP reached the engine")

    # Little Nemo: text-only refusal, then a text turn
    app.select_model("littlenemo")
    yield ("wait", lambda: app.connected, 20, "Little Nemo fixture READY")
    app.add_attachment(png, "image")
    yield ("wait", lambda: all(a["state"] != "checking"
                               for a in app.attachments), 20, "attachment")
    app.input.insert("1.0", "What is in this image?")
    app._input_changed()
    app.on_send()
    _check(not app.busy, "Little Nemo refused the attachment turn")
    _check("text-only" in _chat_text(app), "refusal explained in the chat")
    yield ("sleep", 0.3)
    yield ("shot", "05-littlenemo-text-only-refusal.png")
    for a in list(app.attachments):
        app.remove_attachment(a["id"])
    app.on_send()
    _check(app.busy, "text turn sent to Little Nemo")
    yield ("wait", lambda: not app.busy, 60, "Little Nemo turn DONE")
    yield ("sleep", 0.3)
    yield ("shot", "06-littlenemo-text-turn.png")

    # Appearance: icon picker, shape customizer, theme picker
    app.open_appearance(model_key="supernemo", tab="icons")
    yield ("sleep", 0.5)
    dlg = app._appearance_win
    prev = getattr(dlg, "_preview", None) if dlg else None
    if prev is not None:
        prev.set_state("thinking")
        prev.resume()
    yield ("sleep", 0.4)
    yield ("shot", "08-icon-picker.png")
    yield ("shot", "11-shape-customizer.png")
    if dlg is not None and getattr(dlg, "_show_tab", None):
        dlg._show_tab("themes")
    yield ("sleep", 0.6)
    yield ("shot", "10-theme-picker.png")
    if dlg is not None and getattr(dlg, "_close_appearance", None):
        dlg._close_appearance()
    elif dlg is not None:
        dlg.destroy()
        app._appearance_win = None
    yield ("sleep", 0.3)

    # One screenshot of the main window in each official theme.
    if app.model != "supernemo":
        app.select_model("supernemo")
        yield ("wait", lambda: app.connected or True, 10, "model switch")
    for ident, fname in (
        ("pixel", "12-theme-pixel.png"),
        ("lattice", "13-theme-lattice.png"),
        ("stream", "14-theme-stream.png"),
        ("handshake", "15-theme-handshake.png"),
        ("owl", "16-theme-owl.png"),
    ):
        app.apply_theme(ident, save=False)
        yield ("sleep", 0.45)
        yield ("shot", fname)
    app.apply_theme("pixel", save=False)




def _mascot(app):
    """One shot of the official theme mascot animating on the thinking line."""
    if app.model != "supernemo":
        app.select_model("supernemo")
    yield ("wait", lambda: app.connected, 20, "SuperNemo fixture READY")
    app.input.insert("1.0", "Tell me what you're thinking as you start.")
    app._input_changed()
    app.on_send()
    if not app.busy:
        raise AssertionError("turn did not start")
    yield ("wait",
           lambda: app.cur is not None and app.cur.line.cget("text") != "",
           20, "thinking line")
    # thinking strips run at 10 fps; wait a few frames so the mascot is mid-motion
    yield ("sleep", 0.5)
    yield ("shot", "22-mascot-thinking-line.png")


def _unavailable(app):
    import nemo_chat as proto

    _check(not proto.target_available(app.targets.get("supernemo")),
           "this demo needs SuperNemo without an engine")
    if app.model != "supernemo":
        app.select_model("supernemo")
    yield ("wait", lambda: bool(app.stats.probe), 30, "hardware probe")
    app.input.insert("1.0", "Hello SuperNemo, can you hear me?")
    app._input_changed()
    app.on_send()
    _check(not app.busy, "nothing was sent")
    _check("not available yet" in _chat_text(app), "unavailable notice")
    yield ("sleep", 0.3)
    yield ("shot", "07-supernemo-engine-not-available.png")


def _pickers(app):
    """Appearance dialog shots only (icon picker, shape customizer,
    theme picker). Used to retake 08/10/11 without rerunning the tour."""
    app.open_appearance(model_key="supernemo", tab="icons")
    yield ("sleep", 0.6)
    dlg = app._appearance_win
    _check(dlg is not None, "Appearance dialog opened")
    prev = getattr(dlg, "_preview", None)
    if prev is not None:
        prev.set_state("thinking")
        prev.resume()
    yield ("sleep", 0.5)
    yield ("shot", "08-icon-picker.png")
    yield ("shot", "11-shape-customizer.png")
    if getattr(dlg, "_show_tab", None):
        dlg._show_tab("themes")
    yield ("sleep", 0.7)
    yield ("shot", "10-theme-picker.png")
    if getattr(dlg, "_close_appearance", None):
        dlg._close_appearance()
    yield ("sleep", 0.2)



def _layout(app, what):
    ok, notes = app.layout_ok()
    _check(ok, what + ((": " + "; ".join(notes)) if notes else ""))


def _resize(app):
    """Maximize / restore, minsize, odd sizes, 1080p fullscreen, wrap."""
    app.root.update_idletasks()
    app.root.update()
    # Remember the opening size so restore comes back here.
    app._default_geom = "{0}x{1}+{2}+{3}".format(
        app.root.winfo_width(), app.root.winfo_height(),
        app.root.winfo_x(), app.root.winfo_y())
    mw, mh = app._min_w, app._min_h
    _check(mw >= 640 and mh >= 400, f"minsize is set ({mw}x{mh})")
    # A long message so wrap is visible at every size.
    long = ("The thinking-line icon, the theme banner and every status "
            "dot have to stay crisp while this paragraph wraps instead "
            "of overflowing the chat column. ") * 12
    app.add_user_bubble(long.strip(), [])
    app._chat_write(lambda c: c.insert("end", long + "\n", "assistant"))
    app._relayout(app.root.winfo_width(), app.root.winfo_height())
    yield ("sleep", 0.35)
    _layout(app, "default size layout")
    yield ("shot", "17-default-size.png")

    # Odd and standard sizes. Skip any size larger than the screen.
    sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
    for spec, name in (("1003x701", "odd 1003x701"),
                       ("1280x720", "720p"),
                       ("1366x768", "1366x768")):
        w, h = (int(x) for x in spec.split("x"))
        if w + 20 > sw or h + 40 > sh:
            continue
        # minsize may raise a requested size that is below the floor.
        app.set_window_mode("normal", spec)
        yield ("sleep", 0.3)
        _layout(app, name)
        got_w, got_h = app.root.winfo_width(), app.root.winfo_height()
        if w >= mw and h >= mh:
            # Windows may not land on the exact pixel; allow a small slop
            # and never accept a size below minsize.
            _check(got_w >= mw - 2 and got_h >= mh - 2,
                   f"{name} stayed at or above minsize ({got_w}x{got_h})")

    # An attempt to go below minsize must be rejected by the toolkit.
    app.set_window_mode("normal", f"{max(200, mw // 2)}x{max(150, mh // 2)}")
    yield ("sleep", 0.3)
    got_w, got_h = app.root.winfo_width(), app.root.winfo_height()
    _check(got_w >= mw - 2 and got_h >= mh - 2,
           f"a too-small request was clamped to minsize (got {got_w}x{got_h}, min {mw}x{mh})")

    app.set_window_mode("min")
    yield ("sleep", 0.4)
    _layout(app, "minimum size layout")
    yield ("shot", "19-minimum-size.png")

    app.set_window_mode("maximize")
    yield ("sleep", 0.5)
    _layout(app, "maximized layout")
    # Maximized (or a best-effort fill) should be larger than default.
    _check(app.root.winfo_width() >= min(sw - 40, 1100)
           or app.root.winfo_width() >= app.root.winfo_reqwidth(),
           "maximized window is wide")
    yield ("shot", "18-maximized.png")

    app.set_window_mode("restore")
    yield ("sleep", 0.4)
    _layout(app, "restored layout")

    # 1080p geometry when the display can hold it, then fullscreen.
    if sw >= 1920 and sh >= 1080:
        app.set_window_mode("normal", "1920x1080")
        yield ("sleep", 0.35)
        _layout(app, "1080p geometry")
    app.set_window_mode("fullscreen")
    yield ("sleep", 0.5)
    _layout(app, "fullscreen layout")
    fw, fh = app.root.winfo_width(), app.root.winfo_height()
    _check(fw >= sw - 8 and fh >= sh - 8,
           f"fullscreen covers the screen ({fw}x{fh} vs {sw}x{sh})")
    app.set_window_mode("restore")
    yield ("sleep", 0.3)
    _layout(app, "layout after leaving fullscreen")


def _new_toplevel(app, before):
    import tkinter as tk

    for w in app.root.winfo_children():
        if isinstance(w, tk.Toplevel) and w not in before:
            return w
    return None


def _composer(app):
    """The composer: placeholder, growth to a cap, scrolling, word wrap,
    and the dialogs added with the credits and the CUDA settings."""
    import tkinter as tk
    import nemo_ui

    keys = [k for k in app.targets if k != app.model]
    _check(keys, "the composer demo needs a second model to switch to")
    yield ("sleep", 0.5)
    _check(str(app.input.cget("wrap")) == "word", "input wraps at words")
    ph = app.placeholder.cget("text")
    _check(ph.startswith("Message ") and ph.endswith("\u2026"),
           f"placeholder reads 'Message <model>...' ({ph!r})")
    _check(app.placeholder.winfo_rooty() - app.input.winfo_rooty()
           == nemo_ui.INPUT_PAD, "placeholder starts on the first text line")
    yield ("shot", "20-composer-empty-placeholder.png")
    app.input.insert("1.0", "A short question")
    app._input_changed()
    _check(int(app.input.cget("height")) == 1, "one line stays one line")
    yield ("shot", "21-composer-one-line.png")
    app.input.insert("end", "\nsecond line\nthird line")
    app._input_changed()
    _check(int(app.input.cget("height")) == 3, "three lines grow to three")
    yield ("shot", "22-composer-three-lines.png")
    app.input.insert("end", "".join(f"\nline {i}" for i in range(4, 16)))
    app._input_changed()
    _check(int(app.input.cget("height")) == nemo_ui.INPUT_MAX_LINES,
           "the bar stops growing at the cap and scrolls")
    app.input.see("end")
    yield ("shot", "23-composer-capped-scrolling.png")
    app.input.delete("1.0", "end")
    app._input_changed()
    app.select_model(keys[0])
    yield ("sleep", 0.6)
    ph2 = app.placeholder.cget("text")
    _check(ph2 != ph and ph2.startswith("Message "),
           f"placeholder follows the model ({ph!r} -> {ph2!r})")
    yield ("shot", "24-composer-placeholder-other-model.png")
    before = set(app.root.winfo_children())
    app.open_about()
    dlg = _new_toplevel(app, before)
    _check(dlg is not None, "About dialog opened")
    app._appearance_win = dlg
    yield ("sleep", 0.4)
    dlg.update()
    _check(dlg.winfo_rooty() + dlg.winfo_height() <= dlg.winfo_screenheight(),
           "About dialog fits on the screen")
    yield ("shot", "25-about-and-credits.png")
    dlg.destroy()
    app._appearance_win = None
    before = set(app.root.winfo_children())
    app.open_settings()
    dlg = _new_toplevel(app, before)
    _check(dlg is not None, "settings dialog opened")
    app._appearance_win = dlg
    yield ("sleep", 0.4)
    dlg.update()
    # nothing may be clipped: the whole dialog, with its Apply and Cancel
    # bar, lies on the screen
    _check(dlg.winfo_rooty() >= 0 and dlg.winfo_rooty() + dlg.winfo_height()
           <= dlg.winfo_screenheight(), "settings dialog fits on the screen")
    yield ("shot", "26-settings-dialog.png")
    dlg.destroy()
    app._appearance_win = None


def _boxes(app):
    """Screen rectangles (x0, y0, x1, y1) of the parts that must not overlap."""
    def rect(w):
        w.update_idletasks()
        x, y = w.winfo_rootx(), w.winfo_rooty()
        return (x, y, x + w.winfo_width(), y + w.winfo_height())
    out = {"window": rect(app.root), "composer": rect(app.box)}
    if app.send_btn.winfo_ismapped() and app.send_btn.winfo_width() > 4:
        out["send"] = rect(app.send_btn)
    sp = getattr(app, "stats_frame", None) or getattr(app, "stats_panel", None)
    if sp is not None and sp.winfo_ismapped():
        out["stats"] = rect(sp)
    return out


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0 and \
        min(a[3], b[3]) - max(a[1], b[1]) > 0


def _sizes(app):
    """The window at several sizes with the stats panel open and a
    three-line message: the composer must not run under the stats panel
    or past the window edge, and the send button must stay whole inside
    it. Two sizes below the window's minsize (as Windows Snap or a small
    display can force) are included. Prints one LAYOUT line per size."""
    if not getattr(app, "stats_visible", True):
        app.toggle_stats()
    app.input.insert("1.0", "A three line message\nsecond line\nthird line")
    app._input_changed()
    sw, sh = app.root.winfo_screenwidth(), app.root.winfo_screenheight()
    print(f"SCREEN {sw}x{sh} min {app._min_w}x{app._min_h}", flush=True)
    sizes = [("min", None)] + [("normal", s) for s in
                               ("800x600", "1003x701", "1280x720",
                                "1366x768", "1600x900")] + \
        [("maximize", None), ("forced", "1000x700"), ("forced", "900x600")]
    bad = []
    for mode, spec in sizes:
        if spec:
            w, h = (int(x) for x in spec.split("x"))
            if w + 20 > sw or h + 40 > sh:
                print(f"LAYOUT {spec} skipped (larger than the screen)",
                      flush=True)
                continue
        if mode == "forced":
            app.root.minsize(1, 1)
            app.set_window_mode("normal", spec)
        elif spec:
            app.set_window_mode(mode, spec)
        else:
            app.set_window_mode(mode)
        yield ("sleep", 0.6)
        b = _boxes(app)
        win, comp = b["window"], b["composer"]
        st = b.get("stats")
        tag = spec or mode
        under = bool(st) and _overlap(comp, st)
        past = comp[2] > win[2] + 1 or comp[3] > win[3] + 1
        send = b.get("send")
        send_ok = bool(send) and send[0] >= comp[0] and send[2] <= comp[2] \
            and not (st and _overlap(send, st))
        if not st:
            # Stepping aside on a narrow window is fine; it must come back.
            print(f"LAYOUT {tag} stats panel stepped aside", flush=True)
        print(f"LAYOUT {tag} window {win[2]-win[0]}x{win[3]-win[1]} "
              f"composer {comp} stats {st} under_stats={under} "
              f"past_edge={past} send_ok={send_ok}", flush=True)
        if under or past or not send_ok:
            bad.append(tag)
        yield ("shot", f"30-size-{'forced-' if mode == 'forced' else ''}{tag}.png")
    app.root.minsize(app._min_w, app._min_h)
    app.set_window_mode("normal", "1600x900") if sw >= 1640 else app.set_window_mode("min")
    yield ("sleep", 0.6)
    _check(app.stats_panel.winfo_ismapped(),
           "stats panel is back once the window is wide again")
    app.set_window_mode("restore")
    _check(not bad, f"composer clipped at {bad}")
