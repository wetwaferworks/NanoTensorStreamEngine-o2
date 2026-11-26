#!/usr/bin/env python3
"""nemo_ui.py - the NTSE chat window (Tk).

A clean dark chat layout: a sidebar (new chat, model picker, inputs,
settings), the conversation in the middle with message bubbles and an
input bar with attach buttons, and a live stats panel on the right.
Each reply carries a thinking line: one status line above the answer
that shows what the model is thinking about right now, streamed from
the engine's THINK messages, collapsible to show the whole trace.

Threading: the Tk thread owns every widget. The engine reader thread,
the hardware probe, attachment inspection and VRAM sampling run on
worker threads that only put results on queues; the Tk thread drains
them from an after() loop. No locks.

Launch through nemo_chat.py (see its docstring and PROTOCOL.md).
"""

import os
import queue
import sys
import tempfile
import threading
import time
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox

import nemo_appearance as appearance
import nemo_chat as proto
import nemo_icons as icons
import nemo_media as media
import nemo_shapes as shapes
import nemo_stats as stats
import nemo_themes as themes
import nemo_scale as nscale

BG = "#0e0e10"
SIDEBAR = "#151517"
PANEL = "#141416"
SURFACE = "#1d1d21"
RAISED = "#26262b"
BORDER = "#2e2e34"
SELECT = "#2a2a30"
FG = "#ececf1"
FG2 = "#c9c9d1"
DIM = "#8e8e98"
FAINT = "#5f5f69"
ACCENT = "#f4f4f5"
THINK = "#a5a5b4"
THINK_HOT = "#e4e4ee"
OK = "#4ade80"
WARN = "#f5b041"
WARN_BG = "#2b2210"
ERR = "#ff7a7a"
LAYER_COLORS = {"mamba": "#3d5a99", "mlp": "#34343c", "attention": "#a3622f",
                "moe": "#2f7a5a"}
LAYER_LIT = {"mamba": "#8fb0ff", "mlp": "#9a9aa8", "attention": "#ffb27a",
             "moe": "#6fe0b0"}


def _dpi_aware():
    """Best-effort per-monitor DPI awareness, before Tk() is created."""
    if os.name != "nt":
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
        try:
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
                return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PER_MONITOR
            return
        except Exception:
            pass
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)  # SYSTEM
            return
        except Exception:
            pass
        try:
            user32.SetProcessDPIAware()
        except Exception:
            pass
    except Exception:
        pass


def run_app(targets, model_key, settings_ctx, demo=None, shots=None):
    _dpi_aware()
    app = ChatApp(targets, model_key, settings_ctx or {}, demo, shots)
    app.root.mainloop()
    return app.exit_code


def flatbtn(parent, text, cmd, bg, fg, hover, font, padx=10, pady=5):
    b = tk.Label(parent, text=text, bg=bg, fg=fg, font=font, padx=padx,
                 pady=pady, cursor="hand2")
    b._bg = bg
    b._hover = hover
    b._cmd = cmd
    b.bind("<Button-1>", lambda e: b._cmd())
    b.bind("<Enter>", lambda e: b.configure(bg=b._hover))
    b.bind("<Leave>", lambda e: b.configure(bg=b._bg))
    return b


def publish_palette(pal):
    """Update the module-level colour names so new widgets (and the
    engine-settings dialog) pick up a live theme switch."""
    g = globals()
    g["BG"] = pal["bg"]
    g["SIDEBAR"] = pal["sidebar"]
    g["PANEL"] = pal["panel"]
    g["SURFACE"] = pal["surface"]
    g["RAISED"] = pal["raised"]
    g["BORDER"] = pal["border"]
    g["SELECT"] = pal["select"]
    g["FG"] = pal["fg"]
    g["FG2"] = pal["fg2"]
    g["DIM"] = pal["dim"]
    g["FAINT"] = pal["faint"]
    g["ACCENT"] = pal["accent"]
    g["THINK"] = pal["think"]
    g["THINK_HOT"] = pal.get("think_hot") or pal["think"]
    g["OK"] = pal["ok"]
    g["WARN"] = pal["warn"]
    g["WARN_BG"] = pal["warn_bg"]
    g["ERR"] = pal["error"]


def _pil_transform(im, xf):
    """Apply a motion transform to a Pillow frame, returning an image
    the same size as `im` (transparent padding). Cheap at 32/64 px."""
    from PIL import Image

    w, h = im.size
    out_im = im
    if xf.get("flip_x", 1.0) < 0:
        out_im = out_im.transpose(Image.FLIP_LEFT_RIGHT)
    sx = max(0.06, abs(xf.get("flip_x", 1.0))) * xf.get("scale", 1.0)
    sy = max(0.06, xf.get("squash", 1.0) * xf.get("scale", 1.0))
    nw, nh = max(1, int(round(w * sx))), max(1, int(round(h * sy)))
    if (nw, nh) != out_im.size:
        scaled = nscale.scale_image(out_im, nw, nh, "lanczos3")
        if scaled is not None:
            out_im = scaled
        # If the DLL is missing, leave the native size; the paste
        # centres it. Never resample in Python.
    rot = xf.get("rot", 0.0)
    if abs(rot) > 0.5:
        out_im = out_im.rotate(-rot, resample=Image.Resampling.BILINEAR,
                               expand=False)
    canvas = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ox = (w - out_im.size[0]) // 2 + int(round(xf.get("dx", 0.0)))
    oy = (h - out_im.size[1]) // 2 + int(round(xf.get("dy", 0.0)))
    return icons.alpha_blit(canvas, out_im, (ox, oy))


class IconPlayer:
    """A small Canvas that paints a model's icon or thinking shape and
    animates it from an IconAnimator, driven only by Tk after() timers.
    It stops when settled, when the window is hidden, or when pause()
    is called (turn over, dialog closed). No locks, no threads."""

    def __init__(self, app, parent, size=32, hole_key="bg"):
        self.app = app
        self.size = int(size)
        self.hole_key = hole_key
        px = app.px(self.size)
        self.canvas = tk.Canvas(parent, width=px, height=px, bd=0,
                                highlightthickness=0,
                                bg=app.C.get(hole_key, app.C["bg"]))
        self.anim = icons.IconAnimator(clock=time.monotonic)
        self.procedural = True
        self.cfg = shapes.normalize_shape(None)
        self.pack = None
        self._after = None
        self._visible = True
        self._photo = None
        self._on_click = None
        app._icon_players.append(self)

    def bind_toggle(self, fn):
        self._on_click = fn
        self.canvas.bind("<Button-1>", lambda e: fn())

    def set_appearance(self, cfg, pack=None, procedural=True, model_key="default"):
        self.cfg = shapes.normalize_shape(cfg, model_key)
        self.pack = pack
        self.procedural = bool(procedural)
        self.anim.model_key = model_key
        self.anim.configure(self.cfg, None if procedural else pack)
        self._paint()

    def set_state(self, state):
        self.anim.set_state(state)
        self._paint()
        self._ensure()

    def on_event(self, event, payload=None):
        self.anim.on_event(event, payload)
        self._paint()
        self._ensure()

    def pause(self):
        self._visible = False
        if self._after is not None:
            try:
                self.canvas.after_cancel(self._after)
            except Exception:
                pass
            self._after = None

    def resume(self):
        self._visible = True
        self._ensure()

    def destroy(self):
        self.pause()
        try:
            self.app._icon_players.remove(self)
        except ValueError:
            pass

    def recolor(self):
        self.canvas.configure(bg=self.app.C.get(self.hole_key, self.app.C["bg"]))
        self._paint()

    def _ensure(self):
        if self._after is not None or not self._visible:
            return
        if self.anim.settled or self.anim.state == "idle":
            return
        self._tick()

    def _tick(self):
        self._after = None
        if not self._visible:
            return
        try:
            if not self.canvas.winfo_ismapped():
                return
        except tk.TclError:
            return
        self._paint()
        if self.anim.settled or not self._visible:
            return
        ms = self.anim.interval_ms() or 80
        self._after = self.canvas.after(ms, self._tick)

    def _paint(self):
        state, frame, xf, _settled = self.anim.step()
        px = int(self.canvas.cget("width") or self.app.px(self.size))
        hole = self.canvas.cget("bg")
        if self.procedural or self.pack is None:
            shapes.paint_canvas(self.canvas, self.cfg, state=state,
                                size=px, xf=xf, hole_color=hole)
            return
        im = self.pack.image(state, frame, size=self.size)
        if im is None:
            shapes.paint_canvas(self.canvas, self.cfg, state=state,
                                size=px, xf=xf, hole_color=hole)
            return
        try:
            from PIL import ImageTk

            if im.size != (px, px):
                filt = nscale.filter_for_theme(getattr(self.app, "theme", None))
                scaled = nscale.scale_image(im, px, px, filt)
                if scaled is not None:
                    im = scaled
                # else leave native size (centred below)
            im = _pil_transform(im, xf)
            self._photo = ImageTk.PhotoImage(im)
            self.canvas.delete("shape")
            self.canvas.create_image(px / 2, px / 2, image=self._photo,
                                     tags="shape")
        except Exception:
            shapes.paint_canvas(self.canvas, self.cfg, state=state,
                                size=px, xf=xf, hole_color=hole)


class ThinkLine:
    """The thinking line above one reply."""

    def __init__(self, app):
        self.app = app
        f = self.frame = tk.Frame(app.chat, bg=BG)
        self.spacer = tk.Frame(f, bg=BG, height=1, width=600)
        self.spacer.pack(anchor="w")
        head = self.head = tk.Frame(f, bg=BG, cursor="hand2")
        head.pack(fill="x", anchor="w")
        self.icon = IconPlayer(app, head, size=32, hole_key="bg")
        cfg, pack, proc = app.icon_for(app.model)
        self.icon.set_appearance(cfg, pack, proc, app.model)
        self.icon.set_state("thinking")
        self.icon.canvas.pack(side="left", padx=(0, 8))
        self.title = tk.Label(head, text="Thinking", bg=BG, fg=FG2,
                              font=app.f_small_b)
        self.title.pack(side="left", padx=(0, 8))
        self.chev = tk.Label(head, text="\u25b8", bg=BG, fg=DIM,
                             font=app.f_small)
        self.chev.pack(side="right", padx=(6, 0))
        self.line = tk.Label(head, text="", bg=BG, fg=DIM, anchor="w",
                             font=app.f_small_i)
        self.line.pack(side="left", fill="x", expand=True)
        self.body = tk.Text(f, height=12, wrap="word", bg=SURFACE, fg=THINK,
                            bd=0, highlightthickness=0, padx=12, pady=8,
                            font=app.f_small, state="disabled",
                            cursor="arrow")
        self.body.tag_config("note", foreground=FAINT)
        for w in (head, self.title, self.line, self.chev):
            w.bind("<Button-1>", lambda e: self.toggle())
        self.icon.bind_toggle(self.toggle)
        self.t0 = time.monotonic()
        self.t_answer = None
        self.tail = ""
        self.pieces = 0
        self.chars = 0
        self.answering = False
        self.done = False
        self.expanded = False
        self.max_chars = 80
        self._dirty = False

    def set_width(self, px, char_px):
        self.spacer.configure(width=px)
        self.max_chars = max(24, int(px / max(char_px, 1)) - 22)
        self.body.configure(width=max(30, int(px / max(char_px, 1)) - 4))
        self._dirty = True
        self.refresh()

    def feed(self, piece):
        self.pieces += 1
        self.chars += len(piece)
        self.tail = (self.tail + piece)[-800:]
        self.body.configure(state="normal")
        self.body.insert("end", piece)
        if self.expanded:
            self.body.see("end")
        self.body.configure(state="disabled")
        self._dirty = True

    def note(self, text):
        self.body.configure(state="normal")
        prev = self.body.get("end-2c", "end-1c")
        lead = "" if prev in ("", "\n") else "\n"
        self.body.insert("end", f"{lead}[{text}]\n", "note")
        self.body.configure(state="disabled")
        if not self.chars:
            self.line.configure(text=text)

    def refresh(self):
        """Repaint the headline (throttled by the app's poll loop)."""
        if not self._dirty or self.done:
            return
        self._dirty = False
        if self.answering:
            return
        self.line.configure(
            text=stats.thinking_headline(self.tail, self.max_chars))

    def to_answer(self):
        self.answering = True
        self.t_answer = time.monotonic()
        secs = self.t_answer - self.t0
        if self.chars:
            self.title.configure(text=f"Thought for {secs:.1f}s")
        else:
            self.title.configure(text="Answering")
        self.line.configure(text=stats.thinking_headline(self.tail,
                                                         self.max_chars))

    def finish(self, failed=False, stopped=False):
        if self.done:
            return
        self.done = True
        end = self.t_answer or time.monotonic()
        secs = end - self.t0
        self.icon.on_event("error" if failed else "done")
        if failed and not self.answering:
            self.title.configure(text="Stopped by an error")
        elif self.chars:
            self.title.configure(text=f"Thought for {secs:.1f}s")
        else:
            self.title.configure(text="No thinking trace")
        extra = " \u00b7 stopped" if stopped else ""
        self.line.configure(
            text=(f"{self.pieces} thinking pieces \u00b7 click to "
                  f"{'collapse' if self.expanded else 'expand'}{extra}"
                  if self.chars else extra.strip(" \u00b7")),
            fg=FAINT)

    def pulse(self, phase):
        # The icon player carries the motion now; kept as a no-op so
        # the poll loop's existing call stays harmless.
        return

    def recolor(self):
        C = self.app.C
        for w in (self.frame, self.spacer, self.head, self.title,
                  self.chev, self.line):
            w.configure(bg=C["bg"])
        self.title.configure(fg=C["fg2"])
        self.chev.configure(fg=C["dim"])
        self.line.configure(fg=C["faint"] if self.done else C["dim"])
        self.body.configure(bg=C["surface"], fg=C["think"])
        self.body.tag_config("note", foreground=C["faint"])
        self.icon.hole_key = "bg"
        self.icon.recolor()

    def toggle(self):
        self.expanded = not self.expanded
        if self.expanded:
            self.body.pack(fill="x", anchor="w", pady=(6, 2))
            self.chev.configure(text="\u25be")
            self.body.see("end")
        else:
            self.body.pack_forget()
            self.chev.configure(text="\u25b8")
        if self.done and self.chars:
            self.line.configure(text=(
                f"{self.pieces} thinking pieces \u00b7 click to "
                f"{'collapse' if self.expanded else 'expand'}"))


class ChatApp:
    def __init__(self, targets, model_key, settings_ctx, demo, shots):
        self.settings_ctx = settings_ctx or {}
        self.ini_path = (self.settings_ctx.get("ini_path")
                         or self.settings_ctx.get("config_arg"))
        self.appearance = appearance.load_appearance(self.ini_path)
        # --theme sets the live theme for this session only. It is not
        # written to the INI unless the user later picks a theme in
        # Appearance.
        cli_theme = (self.settings_ctx.get("cli_theme") or "").strip()
        self.theme = themes.get_theme(
            cli_theme or self.appearance.get("theme"),
            appearance.theme_search_dirs(self.ini_path))
        self.C = self.theme["palette"]
        publish_palette(self.C)
        self.targets = appearance.extend_targets(targets, self.ini_path,
                                                 self.appearance)
        self._reload_catalog()
        self.model = model_key if model_key in self.models else (
            proto.MODEL_ORDER[0] if proto.MODEL_ORDER[0] in self.models
            else next(iter(self.models)))
        self.demo = demo
        self.shots = shots
        self.exit_code = 0
        self._icon_players = []
        self._themed = []          # (widget, {option: palette-key})
        self._appearance_win = None
        self._theme_photos = []
        self._lamp_mode = "idle"
        self.events = queue.Queue()
        self.work = queue.Queue()
        self.client = None
        self.gen = 0
        self.ready = {}
        self.connected = False
        self.busy = False
        self.cur = None
        self.last_stats = {}
        self.attachments = []
        self._att_seq = 0
        self.bubbles = []
        self.thinklines = []
        self.photos = []
        self.hero = False
        self.stats = stats.StatsModel()
        self.stats_visible = True
        self._stat_layout = None
        self._stat_labels = {}
        self._t_stats = self._t_pulse = self._t_rss = self._t_vram = 0.0
        self._pulse = False
        self._vram_running = False
        self._has_nvsmi = False
        self.demo_log = []

        root = self.root = tk.Tk()
        root.title("NTSE Chat \u00b7 SuperNemo / Little Nemo")
        root.configure(bg=BG)
        sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
        self.k = self._read_dpi()
        w, h = min(int(1560 * max(1.0, self.k)), sw - 60), min(int(940 * max(1.0, self.k)), sh - 90)
        w, h = max(w, 640), max(h, 400)
        root.geometry(f"{w}x{h}+{max(0, (sw - w) // 2)}+{max(0, (sh - h) // 3)}")
        self._default_geom = f"{w}x{h}"
        self._apply_minsize()
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        root.bind("<Unmap>", self._on_root_unmap)
        root.bind("<Map>", self._on_root_map)
        root.bind("<Configure>", self._on_root_configure)
        self._last_cfg = (0, 0, 0.0)
        self._fonts()
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(0, weight=1)
        self._build_sidebar()
        self._build_main()
        self._build_stats()
        self._apply_window_icon()
        self._apply_theme_chrome()
        threading.Thread(target=self._probe_worker, daemon=True).start()
        self.select_model(self.model, announce=False)
        self.show_hero()
        self.root.after(40, self.poll)
        if demo:
            self.root.after(300, lambda: self._run_demo(demo))

    # ------------------------------------------------------------ build
    def _fonts(self):
        fams = set(tkfont.families(self.root))
        ft = self.theme.get("fonts") or {}

        def pick(want, alias, fallbacks):
            cands = []
            if alias:
                cands.extend(themes.FONT_ALIASES.get(alias, []))
            if want:
                cands.append(want)
                cands.extend(themes.FONT_ALIASES.get(want, []))
            cands.extend(fallbacks)
            seen = set()
            for f in cands:
                if f and f not in seen:
                    seen.add(f)
                    if f in fams:
                        return f
            return fallbacks[-1] if fallbacks else "TkDefaultFont"

        fam = pick(ft.get("family"), ft.get("ui_alias"),
                   ("Segoe UI", "Inter", "Helvetica Neue", "DejaVu Sans",
                    "TkDefaultFont"))
        heading = pick(ft.get("heading") or ft.get("family"),
                       ft.get("heading_alias") or ft.get("ui_alias"),
                       (fam, "Segoe UI", "Inter", "TkDefaultFont"))
        ui_alias = ft.get("ui_alias")
        mono = pick(ft.get("mono"),
                    ui_alias if ui_alias in ("vt", "press", "jb") else None,
                    ("Cascadia Mono", "Consolas", "DejaVu Sans Mono",
                     "TkFixedFont"))
        body, small = int(ft.get("body", 11)), int(ft.get("small", 9))
        title, hero = int(ft.get("title", 13)), int(ft.get("hero", 20))
        specs = {
            "f_body": (fam, body, "normal", "roman"),
            "f_body_b": (fam, body, "bold", "roman"),
            "f_small": (fam, small, "normal", "roman"),
            "f_small_b": (fam, small, "bold", "roman"),
            "f_small_i": (fam, small, "normal", "italic"),
            "f_tiny": (fam, max(7, small - 1), "normal", "roman"),
            "f_tiny_b": (fam, max(7, small - 1), "bold", "roman"),
            "f_title": (heading, title, "bold", "roman"),
            "f_hero": (heading, hero, "bold", "roman"),
            "f_mono": (mono, max(7, small - 1), "normal", "roman"),
        }
        for attr, (family, size, weight, slant) in specs.items():
            font = getattr(self, attr, None)
            if font is None:
                setattr(self, attr, tkfont.Font(family=family, size=size,
                                                weight=weight, slant=slant))
            else:
                font.configure(family=family, size=size, weight=weight,
                               slant=slant)
        self.char_px = self.f_small.measure("0")

    def px(self, n):
        """A layout size in pixels, scaled for the display's DPI."""
        return int(round(n * self.k))

    def _read_dpi(self):
        try:
            return max(0.5, float(self.root.winfo_fpixels("1i")) / 96.0)
        except (tk.TclError, ValueError, TypeError):
            return 1.0

    def _apply_minsize(self):
        """A sensible floor, scaled for DPI, but never larger than the screen."""
        mw = max(640, int(round(980 * self.k)))
        mh = max(400, int(round(600 * self.k)))
        try:
            sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
            mw = min(mw, max(640, sw - 40))
            mh = min(mh, max(400, sh - 60))
        except tk.TclError:
            pass
        self.root.minsize(mw, mh)
        self._min_w, self._min_h = mw, mh
        return mw, mh

    def _build_sidebar(self):
        sb = self.sidebar = tk.Frame(self.root, bg=SIDEBAR, width=self.px(self.theme["spacing"].get("sidebar_width", 236)))
        sb.grid(row=0, column=0, sticky="nsw")
        sb.grid_propagate(False)
        sb.pack_propagate(False)
        brand = tk.Frame(sb, bg=SIDEBAR)
        brand.pack(fill="x", padx=16, pady=(16, 12))
        tk.Label(brand, text="NTSE", bg=SIDEBAR, fg=FG,
                 font=self.f_title).pack(side="left")
        tk.Label(brand, text=" chat", bg=SIDEBAR, fg=DIM,
                 font=self.f_body).pack(side="left")
        nb = self.btn_new = flatbtn(sb, "\uff0b  New chat", self.new_chat,
                     RAISED, FG, SELECT, self.f_body, padx=12, pady=7)
        nb.configure(anchor="w")
        nb.pack(fill="x", padx=12, pady=(0, 14))
        self.btn_new._glyph_action = "new-chat"
        self.btn_new._glyph_text = "  New chat"
        self._side_head(sb, "MODELS")
        self.model_list = tk.Frame(sb, bg=SIDEBAR)
        self.model_list.pack(fill="x")
        self.model_rows = {}
        self._rebuild_model_rows()
        self._side_head(sb, "INPUTS")
        self.inputs_lbl = tk.Label(sb, text="", bg=SIDEBAR, fg=FG2,
                                   font=self.f_small, anchor="w",
                                   justify="left", wraplength=self.px(200))
        self.inputs_lbl.pack(fill="x", padx=18)
        self.frames_lbl = tk.Label(sb, text=media.frame_extractor_note(),
                                   bg=SIDEBAR, fg=FAINT, font=self.f_tiny,
                                   anchor="w", justify="left", wraplength=self.px(200))
        self.frames_lbl.pack(fill="x", padx=18, pady=(4, 0))
        foot = tk.Frame(sb, bg=SIDEBAR)
        foot.pack(side="bottom", fill="x", padx=12, pady=12)
        self.btn_stats = self.btn_appear = self.btn_settings = self.btn_restart = None
        foot_btns = (
            ("Stats panel", self.toggle_stats, None, "btn_stats"),
            ("Appearance\u2026", self.open_appearance, None, "btn_appear"),
            ("Settings\u2026", self.open_settings, "settings", "btn_settings"),
            ("Restart engine", self.restart_engine, None, "btn_restart"),
        )
        for text, cmd, action, attr in foot_btns:
            b = flatbtn(foot, text, cmd, SIDEBAR, FG2, SELECT, self.f_small,
                        padx=8, pady=5)
            b.configure(anchor="w")
            b.pack(fill="x")
            if action:
                b._glyph_action = action
                b._glyph_text = "  " + text
            setattr(self, attr, b)
        self.sidebar_foot_note = tk.Label(
            foot, text="Python front end \u00b7 engines are pure C",
            bg=SIDEBAR, fg=FAINT, font=self.f_tiny,
            anchor="w", justify="left", wraplength=self.px(200))
        self.sidebar_foot_note.pack(fill="x", padx=8, pady=(8, 0))

    def _side_head(self, parent, text):
        tk.Label(parent, text=text, bg=SIDEBAR, fg=FAINT, font=self.f_tiny_b,
                 anchor="w").pack(fill="x", padx=18, pady=(10, 4))

    def _reload_catalog(self):
        self.catalog_list = appearance.catalog(self.ini_path, self.appearance)
        self.models = {m["key"]: m for m in self.catalog_list}

    def info(self, key=None):
        key = self.model if key is None else key
        return self.models.get(key) or {
            "key": key, "label": key, "full": "", "inputs": ("text",),
            "unavailable": f"No engine is configured for {key}.",
            "builtin": False,
        }

    def icon_for(self, key):
        cfg, pack, procedural = appearance.resolve_icon_for(
            self.appearance, key, self.ini_path)
        # Official theme mascots sit on the thinking line when the
        # model has no explicit icon= pack (empty / procedural).
        if procedural or not (cfg.get("icon") or "").strip():
            mascot = icons.theme_mascot_pack(
                (self.theme or {}).get("id"),
                appearance.icon_search_dirs(self.ini_path))
            if mascot is not None:
                return cfg, mascot, False
        return cfg, pack, procedural

    def _rebuild_model_rows(self):
        for rec in list(self.model_rows.values()):
            if rec.get("player"):
                rec["player"].destroy()
            rec["row"].destroy()
        self.model_rows = {}
        parent = self.model_list
        for item in self.catalog_list:
            key = item["key"]
            row = tk.Frame(parent, bg=SIDEBAR, cursor="hand2", padx=8, pady=6)
            row.pack(fill="x", padx=8, pady=1)
            holder = tk.Frame(row, bg=SIDEBAR)
            holder.pack(side="left", padx=(0, 8))
            player = IconPlayer(self, holder, size=28, hole_key="sidebar")
            cfg, pack, proc = self.icon_for(key)
            player.set_appearance(cfg, pack, proc, key)
            player.set_state("idle")
            player.canvas.pack()
            text = tk.Frame(row, bg=SIDEBAR)
            text.pack(side="left", fill="x", expand=True)
            t = tk.Label(text, text=item["label"], bg=SIDEBAR, fg=FG,
                         font=self.f_body_b, anchor="w")
            t.pack(fill="x")
            sub = tk.Label(text, text=item["full"], bg=SIDEBAR, fg=DIM,
                           font=self.f_tiny, anchor="w")
            sub.pack(fill="x")
            st = tk.Label(text, text="", bg=SIDEBAR, fg=FAINT,
                          font=self.f_tiny, anchor="w")
            st.pack(fill="x")
            for wdg in (row, holder, text, t, sub, st, player.canvas):
                wdg.bind("<Button-1>", lambda e, k=key: self.select_model(k))
            # A second click language on the icon itself opens Appearance.
            player.bind_toggle(lambda k=key: self.open_appearance(model_key=k))
            # bind_toggle overwrites the canvas Button-1; also select on
            # a short click-and-open: right-click or double-click for
            # appearance so a single click still selects the model.
            player.canvas.bind("<Button-1>", lambda e, k=key: self.select_model(k))
            player.canvas.bind("<Double-Button-1>",
                               lambda e, k=key: self.open_appearance(model_key=k))
            self.model_rows[key] = {
                "row": row, "holder": holder, "title": t, "sub": sub,
                "status": st, "player": player, "text": text,
            }
        if getattr(self, "inputs_lbl", None) is not None:
            self._refresh_sidebar()

    def _on_root_unmap(self, ev):
        if ev.widget is self.root:
            for p in self._icon_players:
                p.pause()

    def _on_root_map(self, ev):
        if ev.widget is self.root:
            for p in self._icon_players:
                p.resume()

    def _on_root_configure(self, ev):
        if ev.widget is not self.root:
            return
        k = self._read_dpi()
        key = (int(ev.width), int(ev.height), round(k, 3))
        if key == getattr(self, "_last_cfg", None):
            return
        self._last_cfg = key
        if abs(k - self.k) > 0.02:
            self.k = k
            self._on_dpi_change()
        self._relayout(ev.width, ev.height)

    def _on_dpi_change(self):
        """Fonts, pane widths, minsize and cached art follow a DPI change."""
        nscale.clear_cache()
        self._fonts()
        self._apply_minsize()
        try:
            self.sidebar.configure(width=self.px(self.theme["spacing"].get(
                "sidebar_width", 236)))
            self.stats_panel.configure(width=self.px(self.theme["spacing"].get(
                "stats_width", 372)))
        except (tk.TclError, AttributeError):
            pass
        for p in list(self._icon_players):
            try:
                px = self.px(p.size)
                p.canvas.configure(width=px, height=px)
                p._paint()
            except (tk.TclError, AttributeError):
                pass
        try:
            self._apply_window_icon()
            self._apply_theme_chrome()
        except Exception:
            pass
        try:
            self._refresh_sidebar()
            self.render_stats()
        except Exception:
            pass

    def _relayout(self, width=None, height=None):
        """Update wrap-lengths and pane positions after a resize."""
        try:
            sbw = self.sidebar.winfo_width()
            if sbw > 40:
                wrap = max(80, sbw - 36)
                if getattr(self, "inputs_lbl", None):
                    self.inputs_lbl.configure(wraplength=wrap)
                if getattr(self, "frames_lbl", None):
                    self.frames_lbl.configure(wraplength=wrap)
                if getattr(self, "sidebar_foot_note", None):
                    self.sidebar_foot_note.configure(wraplength=wrap)
        except tk.TclError:
            pass
        try:
            if getattr(self, "stats_panel", None) and self.stats_panel.winfo_ismapped():
                pw = self.stats_panel.winfo_width()
                if pw > 40:
                    if getattr(self, "strip_lbl", None):
                        self.strip_lbl.configure(wraplength=max(80, pw - 32))
        except tk.TclError:
            pass
        try:
            mw = self.main.winfo_width()
            if mw > 40:
                tw = self.h_title.winfo_reqwidth() if getattr(self, "h_title", None) else 0
                lw = 0
                if getattr(self, "lamp", None):
                    lw += self.lamp.winfo_reqwidth()
                if getattr(self, "lamp_dot", None):
                    lw += self.lamp_dot.winfo_reqwidth() + 8
                remain = max(80, mw - 40)
                if getattr(self, "h_sub", None):
                    self.h_sub.configure(wraplength=remain)
                if getattr(self, "h_badge", None):
                    self.h_badge.configure(wraplength=remain)
                if getattr(self, "status", None):
                    self.status.configure(wraplength=remain)
                if getattr(self, "hdr_subrow", None):
                    self.hdr_subrow.configure(bg=self.C["bg"])
        except tk.TclError:
            pass
        try:
            if getattr(self, "chat", None):
                self._on_chat_resize(self.chat.winfo_width())
        except tk.TclError:
            pass
        try:
            hint = getattr(self, "composer_hint", None)
            box = getattr(self, "box", None)
            if hint is not None and box is not None:
                used = 16
                for b in (getattr(self, "btn_img", None),
                          getattr(self, "btn_vid", None),
                          getattr(self, "send_btn", None)):
                    if b is not None:
                        used += b.winfo_reqwidth() + 8
                room = box.winfo_width() - used
                if room < self.px(160):
                    hint.pack_forget()
                else:
                    if not hint.winfo_ismapped():
                        hint.pack(side="right", padx=8)
                    hint.configure(wraplength=max(80, room - 8))
        except tk.TclError:
            pass

    def set_window_mode(self, mode, geometry=None):
        """maximize / restore (normal) / min / fullscreen / a WxH string."""
        root = self.root
        try:
            root.attributes("-fullscreen", False)
        except tk.TclError:
            pass
        if mode == "fullscreen":
            try:
                root.attributes("-fullscreen", True)
            except tk.TclError:
                sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                root.geometry(f"{sw}x{sh}+0+0")
        elif mode == "maximize":
            try:
                root.state("zoomed")
            except tk.TclError:
                try:
                    root.attributes("-zoomed", True)
                except tk.TclError:
                    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
                    root.geometry(f"{sw}x{sh}+0+0")
        elif mode == "min":
            try:
                root.state("normal")
            except tk.TclError:
                pass
            try:
                root.attributes("-zoomed", False)
            except tk.TclError:
                pass
            mw, mh = getattr(self, "_min_w", 980), getattr(self, "_min_h", 600)
            root.geometry(f"{mw}x{mh}")
        else:
            try:
                root.state("normal")
            except tk.TclError:
                pass
            try:
                root.attributes("-zoomed", False)
            except tk.TclError:
                pass
            geom = geometry or getattr(self, "_default_geom", None)
            if geom:
                root.geometry(geom)
        try:
            root.update_idletasks()
            root.update()
        except tk.TclError:
            pass
        try:
            self._relayout(root.winfo_width(), root.winfo_height())
        except tk.TclError:
            pass

    def layout_ok(self):
        """(ok, notes) after a resize: panes stay inside the window,
        do not overlap, and the window is not below minsize (except
        when fullscreen)."""
        notes = []
        try:
            root = self.root
            rw, rh = root.winfo_width(), root.winfo_height()
        except tk.TclError:
            return False, ["root gone"]
        fs = False
        try:
            fs = bool(root.attributes("-fullscreen"))
        except tk.TclError:
            pass
        zoomed = False
        try:
            zoomed = (root.state() == "zoomed")
        except tk.TclError:
            pass
        mw = getattr(self, "_min_w", 980)
        mh = getattr(self, "_min_h", 600)
        if not fs and not zoomed:
            if rw < mw - 2:
                notes.append(f"width {rw} is below minsize {mw}")
            if rh < mh - 2:
                notes.append(f"height {rh} is below minsize {mh}")
        boxes = []
        panes = [("sidebar", getattr(self, "sidebar", None)),
                 ("main", getattr(self, "main", None))]
        stats = getattr(self, "stats_panel", None)
        if stats is not None:
            try:
                if stats.winfo_ismapped() and getattr(self, "stats_visible", True):
                    panes.append(("stats", stats))
            except tk.TclError:
                pass
        for name, w in panes:
            if w is None:
                continue
            try:
                x, y = w.winfo_x(), w.winfo_y()
                bw, bh = w.winfo_width(), w.winfo_height()
            except tk.TclError:
                continue
            if bw < 8 or bh < 8:
                notes.append(f"{name} collapsed to {bw}x{bh}")
            if x < -2 or y < -2 or x + bw > rw + 2 or y + bh > rh + 2:
                notes.append(f"{name} at {x},{y} {bw}x{bh} is outside {rw}x{rh}")
            boxes.append((name, x, y, x + bw, y + bh))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                # shrink by 1 px so a shared split line is not an overlap
                if a[1] < b[3] - 1 and b[1] < a[3] - 1 and a[2] < b[4] - 1 and b[2] < a[4] - 1:
                    notes.append(f"{a[0]} overlaps {b[0]}")
        # Composer stays inside the main pane; chat has a usable width.
        try:
            if getattr(self, "composer_outer", None):
                c = self.composer_outer
                mx, my = self.main.winfo_x(), self.main.winfo_y()
                # composer is a child of main; its y+h should fit.
                if c.winfo_y() + c.winfo_height() > self.main.winfo_height() + 4:
                    notes.append("composer overflows the main pane")
            if getattr(self, "chat", None) and self.chat.winfo_width() < 80:
                notes.append(f"chat too narrow ({self.chat.winfo_width()})")
        except tk.TclError:
            pass
        # Bubbles must wrap inside the chat column.
        try:
            col = getattr(self, "_col", None)
            cw = self.chat.winfo_width() if getattr(self, "chat", None) else 0
            if col and cw and col > cw + 40:
                notes.append(f"bubble column {col} exceeds chat {cw}")
            for lbl in getattr(self, "bubbles", []) or []:
                try:
                    wl = int(lbl.cget("wraplength") or 0)
                    if cw and wl > cw:
                        notes.append("a bubble wraplength exceeds the chat")
                        break
                except tk.TclError:
                    pass
        except tk.TclError:
            pass
        return (not notes), notes


    def apply_theme(self, theme_id, save=True):
        """Switch themes live, no restart. Updates the palette, fonts
        and spacing, recolors every open widget, and optionally writes
        the choice to the settings file."""
        theme = themes.get_theme(theme_id, appearance.theme_search_dirs(self.ini_path))
        self.theme = theme
        self.C = theme["palette"]
        self.appearance["theme"] = theme["id"]
        publish_palette(self.C)
        self._fonts()
        self._recolor()
        self._refresh_icons()
        if save and self.ini_path:
            try:
                appearance.save_appearance(self.ini_path, self.appearance)
            except Exception:
                pass
        return theme

    def apply_model_appearance(self, key, cfg, save=True):
        self.appearance = appearance.set_model_cfg(self.appearance, key, cfg)
        if save and self.ini_path:
            try:
                appearance.save_appearance(self.ini_path, self.appearance)
            except Exception:
                pass
        self._refresh_icons(key)

    def _refresh_icons(self, key=None):
        keys = [key] if key else list(self.model_rows)
        for k in keys:
            rec = self.model_rows.get(k)
            if rec and rec.get("player"):
                cfg, pack, proc = self.icon_for(k)
                rec["player"].set_appearance(cfg, pack, proc, k)
                rec["player"].set_state("idle")
        if self.cur is not None and (key is None or key == self.model):
            cfg, pack, proc = self.icon_for(self.model)
            self.cur.icon.set_appearance(cfg, pack, proc, self.model)
            # keep the current animation state
            self.cur.icon._ensure()


    def _load_photo(self, path, size=None, nearest=False):
        """A PhotoImage kept alive on self._theme_photos. size is a
        target (w, h) or a single int for a square. Scaling is done
        in C; if the DLL is missing the prebuilt image is used as-is."""
        if not path or not os.path.isfile(path):
            return None
        filt = nscale.NEAREST if (
            nearest or (self.theme or {}).get("scale") == "nearest"
            or (self.theme or {}).get("id") == "pixel") else nscale.LANCZOS3
        try:
            from PIL import Image, ImageTk
            if size:
                if isinstance(size, int):
                    tw = th = int(size)
                else:
                    tw, th = int(size[0]), int(size[1])
                im = nscale.load_path_scaled(path, tw, th, filt)
            else:
                im = Image.open(path).convert("RGBA")
            if im is None:
                return None
            ph = ImageTk.PhotoImage(im, master=self.root)
        except Exception:
            try:
                ph = tk.PhotoImage(file=path, master=self.root)
            except Exception:
                return None
        self._theme_photos.append(ph)
        return ph

    def _apply_window_icon(self):
        """Use the active theme's .ico (Windows title/taskbar) and PNG
        sizes (iconphoto, including on other platforms)."""
        ico = themes.window_ico_path(self.theme)
        if ico and sys.platform == "win32":
            try:
                self.root.iconbitmap(ico)
            except tk.TclError:
                pass
        photos = []
        for sz in (64, 32, 16):
            path = themes.window_icon_path(self.theme, size=sz)
            ph = self._load_photo(path, size=None)
            if ph is not None:
                photos.append(ph)
        if photos:
            try:
                self.root.iconphoto(True, *photos)
            except tk.TclError:
                try:
                    self.root.iconphoto(True, photos[0])
                except tk.TclError:
                    pass

    def _set_btn_glyph(self, btn, action=None, text=None, size=24):
        if btn is None:
            return
        action = action or getattr(btn, "_glyph_action", None)
        text = text if text is not None else getattr(btn, "_glyph_text", btn.cget("text"))
        path = themes.button_path(self.theme, action, size=size, style="glyph") if action else None
        px = self.px(size)
        ph = self._load_photo(path, size=px) if path else None
        try:
            if ph is not None:
                btn.configure(image=ph, compound="left", text=text or "")
                btn._photo = ph
            else:
                # Restore a text-only fallback (the original marks).
                fallback = text if text else ""
                if action == "send" and not fallback:
                    fallback = "\u2191"
                elif action == "stop" and not fallback:
                    fallback = "\u25a0"
                elif action == "attach-image" and not (text or "").strip().startswith("+"):
                    fallback = "\uff0b" + (text or " Image")
                elif action == "attach-video" and not (text or "").strip().startswith("+"):
                    fallback = "\uff0b" + (text or " Video")
                elif action == "new-chat" and "New chat" in (text or ""):
                    fallback = "\uff0b  New chat"
                btn.configure(image="", text=fallback)
                btn._photo = None
        except tk.TclError:
            pass

    def _apply_theme_chrome(self):
        """Glyph buttons and the status lamp for the active theme."""
        self._theme_photos = [p for p in self._theme_photos if p]
        self._set_btn_glyph(getattr(self, "btn_new", None))
        self._set_btn_glyph(getattr(self, "btn_settings", None))
        self._set_btn_glyph(getattr(self, "btn_img", None))
        self._set_btn_glyph(getattr(self, "btn_vid", None))
        action = "stop" if self.busy else "send"
        self._set_btn_glyph(getattr(self, "send_btn", None), action=action,
                            text="")
        self._set_lamp(getattr(self, "_lamp_mode", "idle"))

    def tint(self, widget, **roles):
        widget._theme_roles = roles
        self._themed.append(widget)
        self._apply_tint(widget)

    def _apply_tint(self, widget):
        roles = getattr(widget, "_theme_roles", None)
        if not roles:
            return
        kw = {}
        for opt, key in roles.items():
            if key in self.C:
                kw[opt] = self.C[key]
        try:
            widget.configure(**kw)
        except tk.TclError:
            pass

    def _recolor(self):
        C = self.C
        try:
            self.root.configure(bg=C["bg"])
        except tk.TclError:
            pass
        # Named major widgets (always present after build).
        for w, key in (
            (getattr(self, "sidebar", None), "sidebar"),
            (getattr(self, "model_list", None), "sidebar"),
            (getattr(self, "main", None), "bg"),
            (getattr(self, "stats_panel", None), "panel"),
            (getattr(self, "chat", None), "bg"),
            (getattr(self, "box", None), "surface"),
            (getattr(self, "chips", None), "surface"),
            (getattr(self, "input", None), "surface"),
            (getattr(self, "placeholder", None), "surface"),
            (getattr(self, "sinner", None), "panel"),
            (getattr(self, "sc", None), "panel"),
            (getattr(self, "strip", None), "panel"),
        ):
            if w is None:
                continue
            try:
                w.configure(bg=C[key])
            except tk.TclError:
                pass
        # Recolor every descendant of the three panes that still carries
        # one of the previous theme colours, using roles when we have
        # them, otherwise a simple bg/fg remap.
        prev = getattr(self, "_last_palette", None)
        remap = {}
        if prev:
            for k, v in prev.items():
                remap[v.lower()] = C.get(k, v)
        for pane in (getattr(self, "sidebar", None),
                     getattr(self, "main", None),
                     getattr(self, "stats_panel", None)):
            if pane is not None:
                self._recolor_tree(pane, remap)
        if getattr(self, "chat", None):
            self.chat.tag_config("assistant", foreground=C["assistant_fg"])
            self.chat.tag_config("sys", foreground=C["faint"])
            self.chat.tag_config("err", foreground=C["error"])
            self.chat.tag_config("divider", foreground=C["faint"])
            self.chat.tag_config("hero", foreground=C["fg"])
            self.chat.tag_config("hero_sub", foreground=C["dim"])
            self.chat.configure(selectbackground=C["select"],
                                insertbackground=C["fg"], fg=C["fg"])
        if getattr(self, "input", None):
            self.input.configure(fg=C["fg"], insertbackground=C["fg"],
                                 bg=C["surface"])
        if getattr(self, "placeholder", None):
            self.placeholder.configure(fg=C["faint"], bg=C["surface"])
        if getattr(self, "h_title", None):
            self.h_title.configure(bg=C["bg"], fg=C["fg"])
            if getattr(self, "hdr_subrow", None):
                self.hdr_subrow.configure(bg=C["bg"])
            self.h_sub.configure(bg=C["bg"], fg=C["dim"])
            self.h_badge.configure(bg=C["bg"], fg=C["warn"])
            self.lamp.configure(bg=C["bg"])
            if getattr(self, "lamp_dot", None):
                self.lamp_dot.configure(bg=C["bg"])
            self.status.configure(bg=C["bg"], fg=C["faint"])
        if getattr(self, "banner", None):
            self.banner.configure(bg=C["warn_bg"])
            self.banner_lbl.configure(bg=C["warn_bg"], fg=C["warn"])
        if getattr(self, "box", None):
            self.box.configure(highlightbackground=C["border"],
                               highlightcolor=C["border"])
        for tl in getattr(self, "thinklines", []):
            tl.recolor()
        for rec in self.model_rows.values():
            sel = rec is self.model_rows.get(self.model)
            # _refresh_sidebar will set the selected colours
            if rec.get("player"):
                rec["player"].recolor()
        for p in self._icon_players:
            p.recolor()
        # spacing
        try:
            self.sidebar.configure(width=self.px(self.theme["spacing"].get(
                "sidebar_width", 236)))
            self.stats_panel.configure(width=self.px(self.theme["spacing"].get(
                "stats_width", 372)))
        except tk.TclError:
            pass
        self._refresh_sidebar()
        self.render_stats()
        self._apply_window_icon()
        self._apply_theme_chrome()
        self._last_palette = dict(C)
        if self._appearance_win is not None:
            try:
                self._appearance_win.configure(bg=C["bg"])
            except tk.TclError:
                pass

    def _recolor_tree(self, widget, remap):
        for child in widget.winfo_children():
            roles = getattr(child, "_theme_roles", None)
            if roles:
                self._apply_tint(child)
            elif remap:
                try:
                    kw = {}
                    for opt in ("bg", "fg", "activebackground", "selectcolor",
                                "highlightbackground", "highlightcolor",
                                "insertbackground"):
                        try:
                            val = str(child.cget(opt)).lower()
                        except tk.TclError:
                            continue
                        if val in remap:
                            kw[opt if opt != "selectcolor" else opt] = remap[val]
                    if kw:
                        child.configure(**kw)
                except tk.TclError:
                    pass
            self._recolor_tree(child, remap)

    def _build_main(self):
        m = self.main = tk.Frame(self.root, bg=BG)
        m.grid(row=0, column=1, sticky="nsew")
        m.grid_columnconfigure(0, weight=1)
        m.grid_rowconfigure(2, weight=1)
        hdr0 = tk.Frame(m, bg=BG)
        hdr0.grid(row=0, column=0, sticky="ew", padx=20, pady=(12, 6))
        hdr = tk.Frame(hdr0, bg=BG)
        hdr.pack(fill="x")
        self.h_title = tk.Label(hdr, text="", bg=BG, fg=FG, font=self.f_title)
        self.h_title.pack(side="left")
        self.lamp = tk.Label(hdr, text="idle", bg=BG, fg=FAINT,
                             font=self.f_small_b)
        self.lamp.pack(side="right")
        self.lamp_dot = tk.Label(hdr, text="\u25cf", bg=BG, fg=FAINT, bd=0)
        self.lamp_dot.pack(side="right", padx=(0, 4))
        # Subtitle and badge live on their own row so a narrow main
        # pane wraps them instead of colliding with the title / lamp.
        self.hdr_subrow = tk.Frame(hdr0, bg=BG)
        self.hdr_subrow.pack(fill="x", pady=(2, 0))
        self.h_sub = tk.Label(self.hdr_subrow, text="", bg=BG, fg=DIM,
                              font=self.f_small, anchor="w", justify="left")
        self.h_sub.pack(side="top", anchor="w", fill="x")
        self.h_badge = tk.Label(self.hdr_subrow, text="", bg=BG, fg=WARN,
                                font=self.f_tiny_b, anchor="w", justify="left")
        self.h_badge.pack(side="top", anchor="w", fill="x")
        self.status = tk.Label(hdr0, text="", bg=BG, fg=FAINT,
                               font=self.f_tiny, anchor="w")
        self.status.pack(fill="x", pady=(2, 0))
        self.banner = tk.Frame(m, bg=WARN_BG)
        self.banner_lbl = tk.Label(self.banner, text="", bg=WARN_BG, fg=WARN,
                                   font=self.f_small, justify="left",
                                   anchor="w", wraplength=760, padx=14,
                                   pady=9)
        self.banner_lbl.pack(fill="x")
        wrap = tk.Frame(m, bg=BG)
        wrap.grid(row=2, column=0, sticky="nsew")
        wrap.grid_columnconfigure(0, weight=1)
        wrap.grid_rowconfigure(0, weight=1)
        self.chat = tk.Text(wrap, bg=BG, fg=FG, bd=0, highlightthickness=0,
                            wrap="word", font=self.f_body, padx=40, pady=10,
                            state="disabled", cursor="arrow",
                            selectbackground="#3a3a44", spacing1=2,
                            spacing3=2)
        self.chat.grid(row=0, column=0, sticky="nsew")
        sbar = tk.Scrollbar(wrap, command=self.chat.yview, width=10)
        sbar.grid(row=0, column=1, sticky="ns")
        self.chat.configure(yscrollcommand=sbar.set)
        c = self.chat
        c.tag_config("assistant", foreground=FG, spacing1=3, spacing3=3,
                     lmargin1=2, lmargin2=2)
        c.tag_config("user_line", justify="right", spacing1=16, spacing3=8)
        c.tag_config("think_line", spacing1=6, spacing3=6)
        c.tag_config("sys", foreground=FAINT, font=self.f_tiny,
                     spacing1=6, spacing3=10)
        c.tag_config("err", foreground=ERR, font=self.f_small, spacing1=6,
                     spacing3=8)
        c.tag_config("divider", foreground=FAINT, font=self.f_tiny,
                     justify="center", spacing1=18, spacing3=14)
        c.tag_config("hero", foreground=FG, font=self.f_hero,
                     justify="center", spacing1=self.px(120), spacing3=8)
        c.tag_config("hero_sub", foreground=DIM, font=self.f_body,
                     justify="center", spacing3=4)
        c.bind("<Configure>", lambda e: self._on_chat_resize(e.width))
        self._build_composer(m)

    def _build_composer(self, m):
        outer = tk.Frame(m, bg=BG)
        outer.grid(row=3, column=0, sticky="ew", padx=40, pady=(4, 14))
        self.composer_outer = outer
        box = self.box = tk.Frame(outer, bg=SURFACE, highlightthickness=1,
                                  highlightbackground=BORDER,
                                  highlightcolor=BORDER)
        box.pack(fill="x")
        self.chips = tk.Frame(box, bg=SURFACE)
        self.chips_warn = tk.Label(box, text="", bg=SURFACE, fg=WARN,
                                   font=self.f_tiny, anchor="w")
        self.input = tk.Text(box, height=2, wrap="word", bg=SURFACE, fg=FG,
                             bd=0, highlightthickness=0, insertbackground=FG,
                             font=self.f_body, padx=14, pady=10, undo=True)
        self.input.pack(fill="x")
        self.placeholder = tk.Label(self.input, text="", bg=SURFACE,
                                    fg=FAINT, font=self.f_body)
        tools = tk.Frame(box, bg=SURFACE)
        tools.pack(fill="x", padx=8, pady=(0, 8))
        self.btn_img = flatbtn(tools, " Image",
                               lambda: self.attach("image"), RAISED, FG2,
                               SELECT, self.f_small, padx=9, pady=4)
        self.btn_img._glyph_action = "attach-image"
        self.btn_img._glyph_text = " Image"
        self.btn_img.pack(side="left", padx=(2, 6))
        self.btn_vid = flatbtn(tools, " Video",
                               lambda: self.attach("video"), RAISED, FG2,
                               SELECT, self.f_small, padx=9, pady=4)
        self.btn_vid._glyph_action = "attach-video"
        self.btn_vid._glyph_text = " Video"
        self.btn_vid.pack(side="left")
        # Voice dictation (parakeet-c, native C speech runtime). Optional:
        # if the module or exe is missing the composer works as before.
        try:
            import nemo_dictation
            self.dictation = nemo_dictation.attach(self, tools)
            self.root.after(1500, lambda: nemo_dictation.prewarm(self.dictation))
        except Exception as _e:  # never break the chat over the mic
            self.dictation = None
            sys.stderr.write(f"dictation unavailable: {_e}\n")
        self.send_btn = flatbtn(tools, "\u2191", self.on_send_or_stop,
                                ACCENT, BG, "#d4d4d8", self.f_body_b,
                                padx=12, pady=3)
        self.send_btn._glyph_action = "send"
        self.send_btn._glyph_text = ""
        self.send_btn.pack(side="right", padx=(6, 2))
        self.composer_hint = tk.Label(
            tools, text="Enter to send \u00b7 Shift+Enter for a new "
            "line \u00b7 Ctrl+V pastes an image", bg=SURFACE, fg=FAINT,
            font=self.f_tiny)
        self.composer_hint.pack(side="right", padx=8)
        self.input.bind("<Return>", self._on_return)
        self.input.bind("<KeyRelease>", lambda e: self._input_changed())
        self.input.bind("<<Paste>>", self._on_paste)
        self.input.bind("<FocusIn>", lambda e: self._input_changed())
        self.input.focus_set()

    def _build_stats(self):
        p = self.stats_panel = tk.Frame(self.root, bg=PANEL, width=self.px(self.theme["spacing"].get("stats_width", 372)))
        p.grid(row=0, column=2, sticky="nse")
        p.grid_propagate(False)
        p.pack_propagate(False)
        hd = tk.Frame(p, bg=PANEL)
        hd.pack(fill="x", padx=16, pady=(16, 2))
        tk.Label(hd, text="Live stats", bg=PANEL, fg=FG,
                 font=self.f_body_b).pack(side="left")
        tk.Label(p, text="Engine-reported unless marked (gui) measured "
                 "here or (probe) read from this machine.", bg=PANEL,
                 fg=FAINT, font=self.f_tiny, justify="left", anchor="w",
                 wraplength=self.px(330)).pack(fill="x", padx=16, pady=(0, 8))
        self.strip = tk.Canvas(p, bg=PANEL, height=self.px(34), highlightthickness=0)
        self.strip.pack(fill="x", padx=16, pady=(0, 4))
        self.strip_lbl = tk.Label(p, text="", bg=PANEL, fg=FAINT,
                                  font=self.f_tiny, anchor="w",
                                  justify="left", wraplength=self.px(330))
        self.strip_lbl.pack(fill="x", padx=16)
        body = tk.Frame(p, bg=PANEL)
        body.pack(fill="both", expand=True, pady=(6, 8))
        self.sc = tk.Canvas(body, bg=PANEL, highlightthickness=0)
        ssb = tk.Scrollbar(body, command=self.sc.yview, width=8)
        self.sc.configure(yscrollcommand=ssb.set)
        ssb.pack(side="right", fill="y")
        self.sc.pack(side="left", fill="both", expand=True)
        self.sinner = tk.Frame(self.sc, bg=PANEL)
        self._sc_win = self.sc.create_window((0, 0), window=self.sinner,
                                             anchor="nw")
        self.sinner.bind("<Configure>", lambda e: self.sc.configure(
            scrollregion=self.sc.bbox("all")))
        self.sc.bind("<Configure>", lambda e: self.sc.itemconfigure(
            self._sc_win, width=e.width))
        for w in (self.sc, self.sinner):
            w.bind("<Enter>", lambda e: self.root.bind_all(
                "<MouseWheel>", self._stats_wheel))
            w.bind("<Leave>", lambda e: self.root.unbind_all("<MouseWheel>"))

    def _stats_wheel(self, e):
        self.sc.yview_scroll(int(-e.delta / 120) or (-1 if e.delta > 0 else 1),
                             "units")

    # ---------------------------------------------------------- layout
    def _on_chat_resize(self, width):
        side = max(self.px(28), (width - self.px(860)) // 2)
        self.chat.configure(padx=side)
        col = width - 2 * side
        for lbl in self.bubbles:
            try:
                lbl.configure(wraplength=int(col * 0.66))
            except tk.TclError:
                pass
        for tl in self.thinklines:
            tl.set_width(col - 8, self.char_px)
        self.banner_lbl.configure(wraplength=max(300, width - 120))
        self.composer_outer.grid_configure(padx=side)
        self._col = col

    def _place_placeholder(self):
        info = self.info(self.model)
        self.placeholder.configure(text=f"Message {info['label']}\u2026")
        if self.input.get("1.0", "end-1c"):
            self.placeholder.place_forget()
        else:
            self.placeholder.place(x=14, y=9)

    def _input_changed(self):
        self._place_placeholder()
        try:
            n = self.input.count("1.0", "end", "displaylines")
        except tk.TclError:
            n = None
        if isinstance(n, (tuple, list)):
            n = n[0] if n else None
        lines = int(n or 1)
        self.input.configure(height=max(2, min(8, lines)))

    # ------------------------------------------------------------ chat
    def _chat_write(self, fn):
        at_bottom = self.chat.yview()[1] > 0.985
        self.chat.configure(state="normal")
        try:
            fn(self.chat)
        finally:
            self.chat.configure(state="disabled")
        if at_bottom:
            self.chat.see("end")

    def show_hero(self):
        info = self.info(self.model)

        def w(c):
            c.delete("1.0", "end")
            c.insert("end", f"{info['label']}\n", "hero")
            avail = proto.target_available(self.targets.get(self.model))
            c.insert("end", info["full"] + "\n", "hero_sub")
            if avail:
                ins = " \u00b7 ".join(i.capitalize() for i in info["inputs"])
                c.insert("end", f"Ask anything. Inputs: {ins}.\n", "hero_sub")
            else:
                c.insert("end", "Engine not available yet: see the notice "
                         "above.\n", "hero_sub")
        self._chat_write(w)
        self.bubbles.clear()
        self.thinklines.clear()
        self.photos = [ph for a in self.attachments for ph in [a.get("photo")]
                       if ph]
        self.hero = True

    def _clear_hero(self):
        if self.hero:
            self._chat_write(lambda c: c.delete("1.0", "end"))
            self.hero = False

    def sys_line(self, text, tag="sys"):
        self._clear_hero()
        self._chat_write(lambda c: c.insert("end", text + "\n", tag))
        self.chat.see("end")  # notices are always brought into view

    def add_user_bubble(self, text, atts):
        self._clear_hero()
        outer = tk.Frame(self.chat, bg=RAISED, padx=14, pady=9)
        if atts:
            row = tk.Frame(outer, bg=RAISED)
            row.pack(anchor="e", pady=(0, 6 if text else 0))
            for a in atts:
                cell = tk.Frame(row, bg=RAISED)
                cell.pack(side="left", padx=(6, 0))
                if a.get("photo_big"):
                    tk.Label(cell, image=a["photo_big"], bg=RAISED).pack()
                    self.photos.append(a["photo_big"])
                meta = a["meta"]
                cap = f"{meta['name']} \u00b7 {media.human_bytes(meta['bytes'])}"
                if meta["type"] == "video":
                    cap += f" \u00b7 {len(a['frames'])} frames sent" \
                        if a["frames"] else " \u00b7 sent by path"
                tk.Label(cell, text=cap, bg=RAISED, fg=DIM,
                         font=self.f_tiny).pack()
        lbl = tk.Label(outer, text=text, bg=RAISED, fg=FG, font=self.f_body,
                       justify="left", anchor="w",
                       wraplength=int(getattr(self, "_col", 760) * 0.66))
        if text:
            lbl.pack(anchor="w")
        self.bubbles.append(lbl)

        def w(c):
            start = c.index("end-1c")
            c.window_create("end", window=outer)
            c.insert("end", "\n")
            c.tag_add("user_line", start, "end-1c")
        self._chat_write(w)
        self.chat.see("end")

    def add_think_line(self):
        tl = ThinkLine(self)
        tl.set_width(getattr(self, "_col", 760) - 8, self.char_px)
        self.thinklines.append(tl)

        def w(c):
            start = c.index("end-1c")
            c.window_create("end", window=tl.frame)
            c.insert("end", "\n")
            c.tag_add("think_line", start, "end-1c")
        self._chat_write(w)
        return tl

    def append_reply(self, piece):
        self._chat_write(lambda c: c.insert("end", piece, "assistant"))

    # ---------------------------------------------------------- models
    def select_model(self, key, announce=True):
        if key not in self.models:
            return
        if self.busy:
            self.sys_line("Wait for the current reply (or press Stop) "
                          "before switching models.", "err")
            return
        changed = key != self.model or not announce
        self.model = key
        info = self.info(key)
        if changed:
            self._disconnect()
            self.stats.reset_engine()
            self.ready = {}
            if announce and not self.hero:
                self.sys_line(f"\u2014 switched to {info['label']} \u00b7 "
                              f"new conversation \u2014", "divider")
            elif announce:
                self.show_hero()
        self.h_title.configure(text=info["label"])
        self.h_sub.configure(text=info["full"])
        self.h_badge.configure(text="")
        avail = proto.target_available(self.targets.get(key))
        if avail:
            self.banner.grid_forget()
            if changed:
                self.connect()
        else:
            self.banner_lbl.configure(text=info["unavailable"])
            self.banner.grid(row=1, column=0, sticky="ew", padx=20,
                             pady=(0, 6))
            self.status.configure(text="not available yet" if key ==
                                  "supernemo" else "no engine configured")
        self._refresh_sidebar()
        self._place_placeholder()
        self._refresh_chips()

    def _refresh_sidebar(self):
        for key, rec in self.model_rows.items():
            row, t, sub, st = rec["row"], rec["title"], rec["sub"], rec["status"]
            sel = key == self.model
            bg = SELECT if sel else SIDEBAR
            for wdg in (row, rec["holder"], t, sub, st):
                wdg.configure(bg=bg)
            if rec.get("player"):
                rec["player"].hole_key = "select" if sel else "sidebar"
                rec["player"].recolor()
            avail = proto.target_available(self.targets.get(key))
            if not avail:
                txt, col = ("\u25cb engine not available yet"
                            if key == "supernemo" else
                            "\u25cb no engine configured"), WARN
            elif sel and self.connected:
                txt, col = "\u25cf connected", OK
            elif sel:
                txt, col = "\u25cc connecting\u2026", DIM
            else:
                txt, col = "\u25cb engine configured", FAINT
            if self.ready.get("fixture") == "1" and sel:
                txt += " \u00b7 test fixture"
            st.configure(text=txt, fg=col)
        info = self.info(self.model)
        if self.connected:
            acc = proto.ready_media(self.ready)
            ins = ["Text"] + [m.capitalize() for m in ("image", "video")
                              if m in acc]
            src = "advertised by the engine"
        else:
            ins = [i.capitalize() for i in info["inputs"]]
            src = "model design; engine not connected"
        if getattr(self, "inputs_lbl", None) is not None:
            self.inputs_lbl.configure(text=" \u00b7 ".join(ins) + f"\n({src})")

    def _disconnect(self):
        if self.client is not None:
            self.client.close()
        self.client = None
        self.connected = False
        self.gen += 1

    def connect(self):
        self._disconnect()
        gen = self.gen
        t = self.targets.get(self.model) or {}
        c = proto.NemoClient(lambda k, p, g=gen: self.events.put((g, k, p)))
        self.client = c
        try:
            if t.get("host") and t.get("port"):
                c.connect_tcp(t["host"], int(t["port"]))
            elif t.get("cmd"):
                c.connect_process(t["cmd"])
            self.status.configure(text="connecting\u2026")
        except Exception as exc:
            self.status.configure(text=f"engine failed to start: {exc}")
            self.sys_line(f"Engine failed to start: {exc}", "err")
            self.client = None
        self._refresh_sidebar()

    def restart_engine(self):
        if self.busy:
            self.sys_line("Stop the current reply before restarting.", "err")
            return
        if not proto.target_available(self.targets.get(self.model)):
            self.sys_line(self.info(self.model)["unavailable"], "err")
            return
        self.sys_line("[engine restarting \u00b7 new conversation]")
        self.stats.reset_engine()
        self.connect()

    def new_chat(self):
        if self.busy:
            self.on_stop()
        try:
            if self.client is not None and self.connected:
                self.client.reset()
        except Exception:
            pass
        self.cur = None
        self.show_hero()

    def toggle_stats(self):
        self.stats_visible = not self.stats_visible
        if self.stats_visible:
            self.stats_panel.grid()
        else:
            self.stats_panel.grid_remove()

    def on_close(self):
        for p in list(self._icon_players):
            p.pause()
        self._disconnect()
        self.root.destroy()

    # ------------------------------------------------------------ send
    def _on_return(self, e):
        if e.state & 0x0001:  # Shift: newline
            return None
        self.on_send()
        return "break"

    def on_send_or_stop(self):
        if self.busy:
            self.on_stop()
        else:
            self.on_send()

    def on_send(self):
        if self.busy:
            return
        text = self.input.get("1.0", "end-1c").strip()
        if any(a["state"] == "checking" for a in self.attachments):
            self.sys_line("Still checking an attachment; try again in a "
                          "moment.", "err")
            return
        atts = [a for a in self.attachments if a["state"] == "ready"]
        if any(a["state"] == "error" for a in self.attachments) \
                and not atts and not text:
            return
        avail = proto.target_available(self.targets.get(self.model))
        ok, msg = proto.check_send(self.model, avail,
                                   self.connected and self.client is not None,
                                   self.ready, text,
                                   [a["meta"] for a in atts],
                                   models=self.models)
        if not ok:
            if msg:
                self.sys_line(msg, "err")
            return
        self.add_user_bubble(text, atts)
        self.cur = self.add_think_line()
        metas = [dict(a["meta"], frames_list=a["frames"]) for a in atts]
        self.stats.turn_begin()
        self.last_stats = {}
        try:
            self.client.send_turn(text, metas)
        except Exception as exc:
            self.cur.finish(failed=True)
            self.sys_line(f"Could not send: {exc}", "err")
            return
        self.busy = True
        self.send_btn.configure(text="\u25a0")
        self._set_btn_glyph(self.send_btn, action="stop", text="")
        self._set_lamp("thinking")
        self.input.delete("1.0", "end")
        self.attachments = [a for a in self.attachments
                            if a["state"] not in ("ready", "error")]
        self._refresh_chips()
        self._input_changed()

    def on_stop(self):
        try:
            if self.client is not None:
                self.client.stop()
        except Exception:
            pass

    def _set_lamp(self, mode):
        self._lamp_mode = mode
        labels = {"thinking": ("thinking", "thinking"),
                  "answering": ("answering", "streaming"),
                  "idle": ("idle", "idle"),
                  "error": ("error", "error"),
                  "expanding": ("expanding", "expanding"),
                  "dtc": ("dense compute", "dtc")}
        txt, state = labels.get(mode, (mode, mode))
        cols = {"thinking": THINK_HOT, "answering": OK, "idle": FAINT,
                "error": ERR, "expanding": self.C.get("accent", DIM),
                "dtc": self.C.get("warn", DIM)}
        col = cols.get(mode, DIM)
        self.lamp.configure(text=txt, fg=col)
        path = themes.status_dot_path(self.theme, state, size=16)
        ph = self._load_photo(path, size=self.px(16)) if path else None
        if getattr(self, "lamp_dot", None) is None:
            return
        try:
            if ph is not None:
                self.lamp_dot.configure(image=ph, text="")
                self.lamp_dot._photo = ph
            else:
                self.lamp_dot.configure(image="", text="\u25cf", fg=col)
                self.lamp_dot._photo = None
        except tk.TclError:
            pass

    def finish_turn(self, failed=False):
        stopped = bool(self.last_stats.get("stopped"))
        if self.cur is not None:
            self.cur.finish(failed=failed, stopped=stopped)
        s = self.last_stats
        if s and not failed:
            bits = [f"{s.get('tokens', 0)} tokens"]
            if "gen_tps" in s:
                bits.append(f"{float(s['gen_tps']):.2f} tok/s generation")
            elif "tps" in s:
                bits.append(f"{float(s['tps']):.2f} tok/s")
            if "ttft_ms" in s:
                bits.append(f"TTFT {float(s['ttft_ms']):.0f} ms")
            if "ms_per_tok" in s:
                bits.append(f"{float(s['ms_per_tok']):.0f} ms/tok")
            if "peak_mb" in s:
                bits.append(f"peak {float(s['peak_mb']):.0f} MB")
            if stopped:
                bits.append("stopped")
            self._chat_write(lambda c: c.insert(
                "end", "\n" + " \u00b7 ".join(bits) + "\n", "sys"))
        else:
            self._chat_write(lambda c: c.insert("end", "\n"))
        self.busy = False
        self.cur = None
        self.send_btn.configure(text="\u2191")
        self._set_btn_glyph(self.send_btn, action="send", text="")
        self._set_lamp("idle")

    # ------------------------------------------------------ attachments
    def attach(self, want):
        if len(self.attachments) >= proto.MAX_ATTACHMENTS:
            self.sys_line(f"At most {proto.MAX_ATTACHMENTS} attachments per "
                          f"turn.", "err")
            return
        exts = media.IMAGE_EXTS if want == "image" else media.VIDEO_EXTS
        pats = " ".join("*" + e for e in exts)
        paths = filedialog.askopenfilenames(
            parent=self.root, title=f"Attach {want}",
            filetypes=[(f"{want.capitalize()} files", pats), ("All", "*.*")])
        for p in paths:
            self.add_attachment(p, want)

    def add_attachment(self, path, want):
        if len(self.attachments) >= proto.MAX_ATTACHMENTS:
            self.sys_line(f"At most {proto.MAX_ATTACHMENTS} attachments per "
                          f"turn; {os.path.basename(path)} was not added.",
                          "err")
            return
        self._att_seq += 1
        a = {"id": self._att_seq, "want": want, "path": path,
             "state": "checking", "meta": None, "frames": [], "note": "",
             "photo": None, "photo_big": None}
        self.attachments.append(a)
        self._refresh_chips()
        threading.Thread(target=self._inspect_worker, args=(a["id"], path,
                                                            want),
                         daemon=True).start()

    def _inspect_worker(self, aid, path, want):
        try:
            meta = media.inspect_attachment(path, want)
            frames, note = ([], "")
            if want == "video":
                frames, note = media.extract_frames(meta, max_frames=8)
            th = media.thumbnail(meta, frames, 44)
            big = media.thumbnail(meta, frames, 168)
            self.work.put(("att_ok", aid, meta, frames, note, th, big))
        except media.AttachmentError as exc:
            self.work.put(("att_err", aid, str(exc)))
        except Exception as exc:  # never let a worker die silently
            self.work.put(("att_err", aid, f"could not read the file ({exc})"))

    def _on_paste(self, e):
        try:
            from PIL import ImageGrab

            clip = ImageGrab.grabclipboard()
        except Exception:
            clip = None
        if clip is None:
            return None
        if isinstance(clip, list):
            for p in clip:
                ext = os.path.splitext(p)[1].lower()
                want = ("image" if ext in media.IMAGE_EXTS and ext != ".gif"
                        else "video" if ext in media.VIDEO_EXTS else None)
                if want:
                    self.add_attachment(p, want)
            return "break"
        try:
            d = tempfile.mkdtemp(prefix="nemo-paste-")
            p = os.path.join(d, f"pasted-{time.strftime('%H%M%S')}.png")
            clip.save(p, "PNG")
            self.add_attachment(p, "image")
        except Exception as exc:
            self.sys_line(f"Could not paste the image: {exc}", "err")
        return "break"

    def remove_attachment(self, aid):
        self.attachments = [a for a in self.attachments if a["id"] != aid]
        self._refresh_chips()

    def _refresh_chips(self):
        for w in self.chips.winfo_children():
            w.destroy()
        if not self.attachments:
            self.chips.pack_forget()
            self.chips_warn.pack_forget()
            return
        self.chips.pack(fill="x", padx=10, pady=(10, 0), before=self.input)
        for a in self.attachments:
            chip = tk.Frame(self.chips, bg=RAISED, padx=6, pady=5)
            chip.pack(side="left", padx=(0, 8))
            if a["photo"] is not None:
                tk.Label(chip, image=a["photo"], bg=RAISED).pack(side="left")
            else:
                tk.Label(chip, text="IMG" if a["want"] == "image" else "VID",
                         bg=SELECT, fg=DIM, font=self.f_tiny_b, width=5,
                         height=2).pack(side="left")
            txt = tk.Frame(chip, bg=RAISED)
            txt.pack(side="left", padx=(6, 4))
            name = os.path.basename(a["path"])
            if len(name) > 26:
                name = name[:12] + "\u2026" + name[-12:]
            tk.Label(txt, text=name, bg=RAISED, fg=FG, font=self.f_small,
                     anchor="w").pack(anchor="w")
            if a["state"] == "checking":
                sub, col = "checking\u2026", DIM
            elif a["state"] == "error":
                sub, col = a["note"], ERR
            else:
                m = a["meta"]
                sub = media.human_bytes(m["bytes"])
                if m.get("w"):
                    sub += f" \u00b7 {m['w']}\u00d7{m['h']}"
                if m["type"] == "video":
                    sub += (f" \u00b7 {len(a['frames'])} frames" if a["frames"]
                            else " \u00b7 no frames (engine decodes)")
                col = DIM
            tk.Label(txt, text=sub, bg=RAISED, fg=col, font=self.f_tiny,
                     anchor="w", wraplength=self.px(260), justify="left").pack(anchor="w")
            x = flatbtn(chip, "\u2715", lambda i=a["id"]: self.remove_attachment(i),
                        RAISED, DIM, SELECT, self.f_small, padx=4, pady=0)
            x.pack(side="left", anchor="n")
        acc = proto.ready_media(self.ready) if self.connected else set(
            i for i in self.info(self.model)["inputs"] if i != "text")
        kinds = {a["want"] for a in self.attachments}
        missing = kinds - acc
        if missing:
            label = self.info(self.model)["label"]
            msg = (f"{label} is text-only: these attachments will not be "
                   f"sent." if not acc else
                   f"This engine does not accept {', '.join(sorted(missing))}.")
            self.chips_warn.configure(text=msg)
            self.chips_warn.pack(fill="x", padx=14, pady=(4, 0),
                                 before=self.input)
        else:
            self.chips_warn.pack_forget()

    # ------------------------------------------------------------ poll
    def poll(self):
        n = 0
        try:
            while n < 600:
                gen, kind, payload = self.events.get_nowait()
                n += 1
                if gen != self.gen:
                    continue  # from a retired engine
                self.handle(kind, payload)
        except queue.Empty:
            pass
        try:
            while True:
                self._handle_work(self.work.get_nowait())
        except queue.Empty:
            pass
        now = time.monotonic()
        if self.cur is not None:
            self.cur.refresh()
        if now - self._t_pulse > 0.45:
            self._t_pulse = now
            self._pulse = not self._pulse
            if self.cur is not None and self.busy:
                self.cur.pulse(self._pulse)
        if now - self._t_rss > 1.0:
            self._t_rss = now
            proc = getattr(self.client, "_proc", None)
            if proc is not None and proc.poll() is None:
                self.stats.engine_rss_mb = stats.process_rss_mb(proc.pid)
        if (self._has_nvsmi and not self._vram_running
                and now - self._t_vram > (4.0 if self.busy else 15.0)):
            self._t_vram = now
            self._vram_running = True
            threading.Thread(target=self._vram_worker, daemon=True).start()
        if now - self._t_stats > 0.25 and self.stats_visible:
            self._t_stats = now
            self.render_stats()
        self.root.after(40, self.poll)

    def handle(self, kind, payload):
        self.stats.on_event(kind, payload)
        if kind == "ready":
            self.ready = payload
            self.connected = True
            r = payload
            bits = [f"{r.get('model', '?')}", f"weights {r.get('weights', '?')}"]
            if "cores" in r:
                bits.append(f"{r.get('threads', '?')}/{r.get('cores', '?')} "
                            f"lanes \u00b7 mode {r.get('mode', '?')} \u00b7 simd "
                            f"{'on' if r.get('simd') == '1' else 'off'}")
            self.status.configure(text="connected \u00b7 " + " \u00b7 ".join(bits))
            self.h_badge.configure(
                text="TEST FIXTURE \u00b7 replies are not model output"
                if r.get("fixture") == "1" else "")
            self._refresh_sidebar()
            self._refresh_chips()
        elif kind == "think":
            if self.cur is not None:
                self.cur.feed(payload)
                self.cur.icon.on_event("think")
        elif kind == "tok":
            if self.cur is None:
                return  # a retired turn (New chat pressed mid-reply)
            if not self.cur.answering:
                self.cur.to_answer()
                self._set_lamp("answering")
            self.cur.icon.on_event("tok")
            self.append_reply(payload)
        elif kind in ("expand", "dtc"):
            if self.cur is not None:
                self.cur.icon.on_event(kind, payload)
        elif kind == "tick":
            if self.busy:
                self._set_lamp(payload.get("mode", "thinking"))
        elif kind == "media_ok":
            if self.cur is not None:
                self.cur.note(f"attachment #{payload.get('id')} accepted by "
                              f"the engine \u00b7 {payload.get('tokens', '?')} "
                              f"tokens")
        elif kind == "media_err":
            msg = (f"Engine refused attachment #{payload.get('id')}: "
                   f"{payload.get('reason', '')}")
            if self.cur is not None:
                self.cur.note(msg)
            self.sys_line(msg, "err")
        elif kind == "stats":
            self.last_stats = payload
        elif kind == "done":
            if self.cur is not None:
                self.cur.icon.on_event("done")
            if self.busy:
                self.finish_turn()
        elif kind == "error":
            self.sys_line(f"Engine error: {payload}", "err")
            if self.cur is not None:
                self.cur.icon.on_event("error")
            if self.busy:
                self.finish_turn(failed=True)
        elif kind == "closed":
            self.connected = False
            self.status.configure(text="disconnected")
            if self.busy:
                self.sys_line("The engine closed the connection.", "err")
                self.finish_turn(failed=True)
            self._refresh_sidebar()

    def _handle_work(self, item):
        kind = item[0]
        if kind == "hw":
            self.stats.probe = item[1]
            self._has_nvsmi = "cuda" in item[1]
            if "vram_used_mb" in item[1]:
                self.stats.vram_sample = item[1]["vram_used_mb"]
        elif kind == "vram":
            self._vram_running = False
            if item[1] is not None:
                self.stats.vram_sample = item[1]
        elif kind in ("att_ok", "att_err"):
            a = next((x for x in self.attachments if x["id"] == item[1]), None)
            if a is None:
                return  # removed while being checked
            if kind == "att_ok":
                _, _, meta, frames, note, th, big = item
                a.update(state="ready", meta=meta, frames=frames, note=note)
                try:
                    from PIL import ImageTk

                    if th is not None:
                        a["photo"] = ImageTk.PhotoImage(th)
                    if big is not None:
                        a["photo_big"] = ImageTk.PhotoImage(big)
                except Exception:
                    pass
            else:
                a.update(state="error", note=item[2])
            self._refresh_chips()

    def _probe_worker(self):
        try:
            self.work.put(("hw", stats.probe_hardware()))
        except Exception:
            self.work.put(("hw", {}))

    def _vram_worker(self):
        try:
            g = stats.nvidia_smi()
            self.work.put(("vram", g[0]["vram_used_mb"] if g else None))
        except Exception:
            self.work.put(("vram", None))

    # ----------------------------------------------------------- stats
    def render_stats(self):
        rows = self.stats.rows()
        layout = tuple((sec, tuple(k for k, _ in items)) for sec, items in rows)
        if layout != self._stat_layout:
            for w in self.sinner.winfo_children():
                w.destroy()
            self._stat_labels = {}
            for sec, items in rows:
                tk.Label(self.sinner, text=sec.upper(), bg=PANEL, fg=FAINT,
                         font=self.f_tiny_b, anchor="w").pack(
                             fill="x", padx=16, pady=(10, 3))
                grid = tk.Frame(self.sinner, bg=PANEL)
                grid.pack(fill="x", padx=16)
                grid.grid_columnconfigure(1, weight=1)
                for r, (k, v) in enumerate(items):
                    tk.Label(grid, text=k, bg=PANEL, fg=DIM, font=self.f_tiny,
                             anchor="nw", justify="left", wraplength=self.px(120)).grid(
                                 row=r, column=0, sticky="nw", pady=1)
                    lv = tk.Label(grid, text=v, bg=PANEL, fg=FG2,
                                  font=self.f_tiny, anchor="nw",
                                  justify="left", wraplength=self.px(196))
                    lv.grid(row=r, column=1, sticky="nw", padx=(8, 0), pady=1)
                    self._stat_labels[(sec, k)] = lv
            self._stat_layout = layout
        else:
            for sec, items in rows:
                for k, v in items:
                    lv = self._stat_labels.get((sec, k))
                    if lv is not None and lv.cget("text") != v:
                        lv.configure(text=v)
        self.draw_strip()

    def draw_strip(self):
        c = self.strip
        c.delete("all")
        try:
            n = int(self.ready.get("layers", 0))
        except (TypeError, ValueError):
            n = 0
        if n <= 0:
            self.strip_lbl.configure(text="layers: waiting for the engine")
            return
        n = min(n, 512)
        w = max(c.winfo_width(), 200)
        cw = w / n
        lyr = self.stats.layer
        staged = int(self.stats.tick.get("staged", 0) or 0) if self.busy else 0
        known = self.stats.layer_types
        use_pattern = n == len(proto.LAYER_TYPES) and self.model == "littlenemo"
        for i in range(n):
            t = (proto.LAYER_TYPES[i] if use_pattern else known.get(i, "?"))
            hot = self.busy and lyr is not None and i == lyr
            fill = (LAYER_LIT if hot else LAYER_COLORS).get(t, "#2a2a30")
            y0 = self.px(4)
            c.create_rectangle(i * cw + 0.5, y0, (i + 1) * cw - 0.5,
                               self.px(24),
                               fill=fill, outline="#ffffff" if hot else "",
                               width=1)
            if self.busy and lyr is not None and lyr < i <= lyr + staged:
                c.create_line(i * cw + 1, self.px(29), (i + 1) * cw - 1,
                              self.px(29),
                              fill=ACCENT, width=2)
        self.strip_lbl.configure(
            text=f"{n} layers \u00b7 blue mamba \u00b7 grey mlp \u00b7 orange "
                 f"attention \u00b7 outlined = computing \u00b7 underline = "
                 f"staged")

    # -------------------------------------------------------- settings
    def open_settings(self):
        open_settings_dialog(self)

    def open_appearance(self, model_key=None, tab="icons"):
        open_appearance_dialog(self, model_key=model_key or self.model, tab=tab)

    # ------------------------------------------------------------ demo
    def _run_demo(self, name):
        import nemo_demo

        gen = nemo_demo.script(self, name)
        root = self.root

        def fail(msg):
            self.demo_log.append("FAIL " + msg)
            print("DEMO FAIL:", msg, flush=True)
            self.exit_code = 1
            root.after(200, self.on_close)

        def step():
            try:
                cmd = next(gen)
            except StopIteration:
                print("DEMO OK" if self.exit_code == 0 else "DEMO FAILED",
                      flush=True)
                root.after(200, self.on_close)
                return
            except Exception as exc:  # a broken step fails the demo
                fail(f"step raised {exc!r}")
                return
            kind = cmd[0]
            if kind == "sleep":
                root.after(int(cmd[1] * 1000), step)
            elif kind == "wait":
                cond, timeout, what = cmd[1], cmd[2], cmd[3]
                deadline = time.monotonic() + timeout

                def check():
                    try:
                        good = cond()
                    except Exception:
                        good = False
                    if good:
                        step()
                    elif time.monotonic() > deadline:
                        fail(f"timed out waiting for {what}")
                    else:
                        root.after(80, check)
                check()
            elif kind == "shot":
                self._shoot(cmd[1], step)
            else:
                fail(f"unknown demo step {kind}")
        step()

    def _shoot(self, name, then):
        if not self.shots:
            self.root.after(50, then)
            return
        os.makedirs(self.shots, exist_ok=True)
        root = self.root
        root.deiconify()
        root.lift()
        # If the Appearance dialog is open, do not raise the main
        # window over it (topmost on the root would hide the picker).
        extra = getattr(self, "_appearance_win", None)
        try:
            if extra is None or not extra.winfo_exists():
                extra = None
        except tk.TclError:
            extra = None
        if extra is not None:
            try:
                extra.deiconify()
                extra.lift()
                extra.attributes("-topmost", True)
            except tk.TclError:
                extra = None
        if extra is None:
            root.attributes("-topmost", True)
        self.render_stats()
        root.update_idletasks()
        root.update()
        if extra is not None:
            try:
                extra.update_idletasks()
                extra.update()
            except tk.TclError:
                pass

        def grab():
            try:
                from PIL import ImageGrab

                x, y = root.winfo_rootx(), root.winfo_rooty()
                w, h = root.winfo_width(), root.winfo_height()
                x2, y2 = x + w, y + h
                if extra is not None:
                    try:
                        ex, ey = extra.winfo_rootx(), extra.winfo_rooty()
                        ew, eh = extra.winfo_width(), extra.winfo_height()
                        x, y = min(x, ex), min(y, ey)
                        x2, y2 = max(x2, ex + ew), max(y2, ey + eh)
                    except tk.TclError:
                        pass
                img = ImageGrab.grab(bbox=(x, y, x2, y2), all_screens=True)
                path = os.path.join(self.shots, name)
                img.save(path)
                print("SHOT", path, flush=True)
            except Exception as exc:
                print("SHOT FAILED", name, exc, flush=True)
            try:
                if extra is not None:
                    extra.attributes("-topmost", False)
            except tk.TclError:
                pass
            try:
                root.attributes("-topmost", False)
            except tk.TclError:
                pass
            then()
        root.after(450, grab)


def open_appearance_dialog(app, model_key=None, tab="icons"):
    """Theme picker, icon picker and shape customizer. Changes apply
    live (no restart) and are written to the settings file. A second
    open raises the existing window instead of stacking another."""
    C = app.C
    if app._appearance_win is not None:
        try:
            if app._appearance_win.winfo_exists():
                app._appearance_win.lift()
                app._appearance_win._show_tab(tab)
                if model_key:
                    app._appearance_win._select_model(model_key)
                return
        except tk.TclError:
            app._appearance_win = None

    dlg = tk.Toplevel(app.root)
    app._appearance_win = dlg
    dlg.title("Appearance")
    dlg.configure(bg=C["bg"])
    dlg.transient(app.root)
    dlg.geometry(f"{app.px(780)}x{app.px(640)}")
    dlg.minsize(app.px(700), app.px(560))

    state = {
        "tab": tab if tab in ("themes", "icons", "shapes") else "icons",
        "model": model_key if model_key in app.models else app.model,
        "preview": None,
        "save_after": None,
    }

    hdr = tk.Frame(dlg, bg=C["bg"])
    hdr.pack(fill="x", padx=16, pady=(14, 8))
    tk.Label(hdr, text="Appearance", bg=C["bg"], fg=C["fg"],
             font=app.f_title).pack(side="left")
    tk.Label(hdr, text="Changes apply immediately and are saved to the "
             "settings file. No restart.", bg=C["bg"], fg=C["dim"],
             font=app.f_tiny).pack(side="left", padx=12, pady=(4, 0))

    tabs = tk.Frame(dlg, bg=C["bg"])
    tabs.pack(fill="x", padx=16)
    pages = {}
    tab_btns = {}

    def show_tab(name):
        state["tab"] = name if name != "shapes" else "icons"
        for n, page in pages.items():
            if n == state["tab"]:
                page.pack(fill="both", expand=True, padx=16, pady=10)
            else:
                page.pack_forget()
        for n, b in tab_btns.items():
            on = (n == name) or (name == "shapes" and n == "icons")
            b.configure(bg=C["select"] if on else C["raised"],
                        fg=C["fg"] if on else C["fg2"])
        if name == "shapes" and state.get("preview"):
            state["preview"].set_state("thinking")
            state["preview"].resume()

    dlg._show_tab = show_tab

    for name, label in (("themes", "Themes"),
                        ("icons", "Icons & shapes")):
        b = flatbtn(tabs, label, lambda n=name: show_tab(n),
                    C["raised"], C["fg2"], C["select"], app.f_small_b,
                    padx=14, pady=6)
        b.pack(side="left", padx=(0, 8))
        tab_btns[name] = b

    body = tk.Frame(dlg, bg=C["bg"])
    body.pack(fill="both", expand=True)

    # ---------------- themes ----------------
    tpage = tk.Frame(body, bg=C["bg"])
    pages["themes"] = tpage
    tk.Label(tpage, text="Official CloudAtlas pack (gui/theme-pack/theme.json). "
             "Default is Pixel Nemo. All five are selectable; a themes/ folder "
             "beside the settings file can still add more \u2014 see THEMES.md.",
             bg=C["bg"], fg=C["dim"],
             font=app.f_small, wraplength=720, justify="left",
             anchor="w").pack(fill="x", pady=(0, 10))
    twrap = tk.Frame(tpage, bg=C["bg"])
    twrap.pack(fill="both", expand=True)
    tcanvas = tk.Canvas(twrap, bg=C["bg"], highlightthickness=0)
    tscroll = tk.Scrollbar(twrap, command=tcanvas.yview, width=10)
    tcanvas.configure(yscrollcommand=tscroll.set)
    tscroll.pack(side="right", fill="y")
    tcanvas.pack(side="left", fill="both", expand=True)
    tlist = tk.Frame(tcanvas, bg=C["bg"])
    twid = tcanvas.create_window((0, 0), window=tlist, anchor="nw")
    def _sync_themes(_e=None):
        tcanvas.configure(scrollregion=tcanvas.bbox("all") or (0, 0, 0, 0))
        try:
            tcanvas.itemconfigure(twid, width=tcanvas.winfo_width())
        except tk.TclError:
            pass
    tlist.bind("<Configure>", _sync_themes)
    tcanvas.bind("<Configure>", lambda e: tcanvas.itemconfigure(twid, width=e.width))
    def _wheel(ev):
        delta = getattr(ev, "delta", 0) or 0
        if getattr(ev, "num", None) == 5 or delta < 0:
            tcanvas.yview_scroll(1, "units")
        else:
            tcanvas.yview_scroll(-1, "units")
        return "break"
    def _bind_wheel(_e=None):
        tcanvas.bind_all("<MouseWheel>", _wheel)
        tcanvas.bind_all("<Button-4>", _wheel)
        tcanvas.bind_all("<Button-5>", _wheel)
    def _unbind_wheel(_e=None):
        tcanvas.unbind_all("<MouseWheel>")
        tcanvas.unbind_all("<Button-4>")
        tcanvas.unbind_all("<Button-5>")
    for w in (tcanvas, tlist):
        w.bind("<Enter>", _bind_wheel)
        w.bind("<Leave>", _unbind_wheel)
    dlg._unbind_theme_wheel = _unbind_wheel
    theme_cards = []
    dlg._photos = []

    def paint_themes():
        for w in tlist.winfo_children():
            w.destroy()
        theme_cards.clear()
        found = themes.discover_themes(appearance.theme_search_dirs(app.ini_path))
        for th in found:
            if "not valid JSON" in ";".join(th.get("notes") or []):
                continue
            selected = th["id"] == app.theme["id"]
            card = tk.Frame(tlist, bg=C["surface"], highlightthickness=2,
                            highlightbackground=C["accent"] if selected
                            else C["border"], cursor="hand2")
            card.pack(fill="x", pady=5)
            banner_path = themes.banner_path(th)
            if banner_path:
                try:
                    from PIL import Image, ImageTk
                    # 4:1 banner, ~320x80 in the card, scaled in C.
                    bw = app.px(320)
                    src = Image.open(banner_path)
                    bh = max(1, int(round(bw * src.size[1] / max(1, src.size[0]))))
                    filt = nscale.filter_for_theme(th)
                    bim = nscale.scale_image(src.convert("RGBA"), bw, bh, filt)
                    if bim is None:
                        # No Python resample: skip a banner we cannot size
                        # to the card (the 1200x300 source would blow the
                        # dialog). The swatches and title still identify
                        # the theme.
                        bim = None
                    if bim is not None:
                        bph = ImageTk.PhotoImage(bim, master=dlg)
                        dlg._photos.append(bph)
                        bl = tk.Label(card, image=bph, bg=C["surface"], bd=0)
                        bl.pack(fill="x")
                        bl._photo = bph
                except Exception:
                    pass
            inner = tk.Frame(card, bg=C["surface"])
            inner.pack(fill="x", padx=12, pady=10)
            title_bits = th.get("name") or th["id"]
            if th.get("subtitle"):
                title_bits = title_bits + "  \u00b7  " + th["subtitle"]
            tk.Label(inner, text=title_bits, bg=C["surface"],
                     fg=C["fg"], font=app.f_body_b, anchor="w").pack(
                         side="left")
            sub = th.get("author") or ""
            if th.get("default") or th["id"] == themes.default_theme_id():
                sub = (sub + "  \u00b7  default").strip(" \u00b7")
            tk.Label(inner, text=sub, bg=C["surface"], fg=C["dim"],
                     font=app.f_tiny, anchor="w").pack(side="left", padx=10)
            sw = tk.Frame(inner, bg=C["surface"])
            sw.pack(side="right")
            pal = th["palette"]
            for key in ("accent", "accent2", "bg", "surface", "fg"):
                tk.Frame(sw, bg=pal[key], width=22, height=22,
                         highlightthickness=1,
                         highlightbackground=C["border"]).pack(
                             side="left", padx=2)
            def pick(ev, ident=th["id"]):
                app.apply_theme(ident, save=True)
                # Recolor this dialog too, then rebuild the cards so the
                # selected outline follows.
                try:
                    dlg.configure(bg=app.C["bg"])
                except tk.TclError:
                    pass
                # The dialog is mid-rebuild; defer so apply_theme finishes.
                dlg.after(10, lambda: (refresh_dialog_colors(), paint_themes()))
            def _bind_tree(w, fn):
                w.bind("<Button-1>", fn)
                for ch in w.winfo_children():
                    _bind_tree(ch, fn)
            _bind_tree(card, pick)
            theme_cards.append(card)
            tlist.update_idletasks()
            _sync_themes()

    def refresh_dialog_colors():
        C2 = app.C
        for w in (dlg, hdr, tabs, body, tpage):
            try:
                w.configure(bg=C2["bg"])
            except tk.TclError:
                pass
        # labels in hdr
        for child in hdr.winfo_children():
            try:
                child.configure(bg=C2["bg"],
                                fg=C2["fg"] if child.cget("text") == "Appearance"
                                else C2["dim"])
            except tk.TclError:
                pass

    paint_themes()

    # ---------------- icons & shapes ----------------
    ipage = tk.Frame(body, bg=C["bg"])
    pages["icons"] = ipage

    top = tk.Frame(ipage, bg=C["bg"])
    top.pack(fill="x")
    tk.Label(top, text="Model", bg=C["bg"], fg=C["fg2"],
             font=app.f_small).pack(side="left")
    model_var = tk.StringVar(value=state["model"])
    model_menu = tk.OptionMenu(top, model_var, *list(app.models.keys()))
    model_menu.configure(bg=C["surface"], fg=C["fg"],
                         activebackground=C["select"],
                         highlightthickness=0, relief="flat")
    model_menu.pack(side="left", padx=8)
    # rewrite labels to the model's display name
    mmenu = model_menu["menu"]
    mmenu.delete(0, "end")
    for k, info in app.models.items():
        mmenu.add_command(label=info["label"],
                          command=lambda v=k: model_var.set(v))

    src_var = tk.StringVar(value="shape")
    src_row = tk.Frame(ipage, bg=C["bg"])
    src_row.pack(fill="x", pady=(10, 6))
    tk.Radiobutton(src_row, text="Thinking shape  (procedural)",
                   variable=src_var, value="shape", bg=C["bg"], fg=C["fg"],
                   selectcolor=C["surface"], activebackground=C["bg"],
                   font=app.f_small, highlightthickness=0).pack(side="left")
    tk.Radiobutton(src_row, text="PNG icon pack  (overrides the shape)",
                   variable=src_var, value="pack", bg=C["bg"], fg=C["fg"],
                   selectcolor=C["surface"], activebackground=C["bg"],
                   font=app.f_small, highlightthickness=0).pack(side="left",
                                                                padx=16)

    mid = tk.Frame(ipage, bg=C["bg"])
    mid.pack(fill="both", expand=True, pady=(4, 0))
    mid.grid_columnconfigure(0, weight=1)
    mid.grid_columnconfigure(1, weight=1)

    # left: shape grid + pack list
    left = tk.Frame(mid, bg=C["bg"])
    left.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
    tk.Label(left, text="Base shape", bg=C["bg"], fg=C["faint"],
             font=app.f_tiny_b, anchor="w").pack(fill="x")
    grid = tk.Frame(left, bg=C["bg"])
    grid.pack(anchor="w", pady=4)
    shape_btns = {}
    shape_previews = {}

    pack_box = tk.Frame(left, bg=C["bg"])
    pack_box.pack(fill="both", expand=True, pady=(12, 0))
    tk.Label(pack_box, text="Icon packs  (double-click a built-in, or Browse)",
             bg=C["bg"], fg=C["faint"], font=app.f_tiny_b,
             anchor="w").pack(fill="x")
    pack_list = tk.Listbox(pack_box, bg=C["surface"], fg=C["fg"],
                           selectbackground=C["select"],
                           highlightthickness=0, bd=0, font=app.f_small,
                           height=7)
    pack_list.pack(fill="both", expand=True, pady=4)
    pack_btns = tk.Frame(pack_box, bg=C["bg"])
    pack_btns.pack(fill="x")
    pack_ids = []  # parallel to listbox rows

    def fill_packs():
        pack_list.delete(0, "end")
        pack_ids.clear()
        packs = icons.discover_packs(appearance.icon_search_dirs(app.ini_path))
        packs.extend(themes.list_thinking_packs(
            appearance.theme_search_dirs(app.ini_path)))
        for p in packs:
            pack_ids.append(p["id"])
            pack_list.insert("end", p["name"])

    fill_packs()

    # right: preview + controls
    right = tk.Frame(mid, bg=C["bg"])
    right.grid(row=0, column=1, sticky="nsew")
    tk.Label(right, text="Live preview", bg=C["bg"], fg=C["faint"],
             font=app.f_tiny_b, anchor="w").pack(fill="x")
    prev_holder = tk.Frame(right, bg=C["surface"], width=app.px(120),
                           height=app.px(120))
    prev_holder.pack(pady=8)
    prev_holder.pack_propagate(False)
    preview = IconPlayer(app, prev_holder, size=64, hole_key="surface")
    preview.canvas.place(relx=0.5, rely=0.5, anchor="center")
    state["preview"] = preview
    dlg._preview = preview
    state_row = tk.Frame(right, bg=C["bg"])
    state_row.pack(fill="x", pady=(0, 8))
    for st in icons.STATES:
        flatbtn(state_row, st, lambda n=st: preview.set_state(n),
                C["raised"], C["fg2"], C["select"], app.f_tiny,
                padx=6, pady=3).pack(side="left", padx=2)

    # colour / face / size / motion / speed
    ctrls = tk.Frame(right, bg=C["bg"])
    ctrls.pack(fill="x", pady=4)
    fill_var = tk.StringVar()
    out_var = tk.StringVar()
    face_var = tk.BooleanVar(value=True)
    size_var = tk.IntVar(value=32)
    motion_var = tk.StringVar(value="bob")
    speed_var = tk.DoubleVar(value=1.0)
    shape_var = tk.StringVar(value="blob")
    icon_var = tk.StringVar(value="")
    note = tk.Label(right, text="", bg=C["bg"], fg=C["dim"],
                    font=app.f_tiny, wraplength=300, justify="left",
                    anchor="w")
    note.pack(fill="x", pady=(8, 0))

    def current_cfg():
        return {
            "shape": shape_var.get(),
            "fill": fill_var.get(),
            "outline": out_var.get(),
            "face": bool(face_var.get()),
            "size": int(size_var.get()),
            "motion": motion_var.get(),
            "speed": float(speed_var.get()),
            "icon": icon_var.get() if src_var.get() == "pack" else "",
        }

    def apply_live(save=True):
        cfg = current_cfg()
        key = model_var.get()
        app.apply_model_appearance(key, cfg, save=save)
        cfg, pack, proc = app.icon_for(key)
        preview.set_appearance(cfg, pack, proc, key)
        if preview.anim.state == "idle":
            preview.set_state("thinking")
        else:
            preview._ensure()
        if proc and src_var.get() == "pack" and icon_var.get():
            note.configure(text="Pack could not be loaded: "
                           + (pack.reason or "fallback")
                           + ". The thinking shape is used instead.",
                           fg=app.C["warn"])
        elif src_var.get() == "pack" and not proc:
            note.configure(text="Using the PNG pack (it overrides the "
                           "thinking shape until you clear it).",
                           fg=app.C["dim"])
        else:
            note.configure(text="Using the thinking shape. Pick a PNG pack "
                           "to override it.", fg=app.C["dim"])
        # Refresh the little shape-grid outlines
        for name, holder in shape_previews.items():
            try:
                holder.configure(highlightbackground=app.C["accent"]
                                 if name == shape_var.get() else app.C["border"])
            except tk.TclError:
                pass

    def load_model(key):
        state["model"] = key
        cfg, pack, proc = app.icon_for(key)
        shape_var.set(cfg["shape"])
        fill_var.set(cfg["fill"])
        out_var.set(cfg["outline"])
        face_var.set(bool(cfg["face"]))
        size_var.set(int(cfg["size"]))
        motion_var.set(cfg["motion"])
        speed_var.set(float(cfg["speed"]))
        icon_var.set(cfg.get("icon") or "")
        src_var.set("pack" if cfg.get("icon") and not proc else "shape")
        # highlight pack in the list
        if cfg.get("icon"):
            try:
                pack_list.selection_clear(0, "end")
                pack_list.selection_set(pack_ids.index(cfg["icon"]))
            except ValueError:
                pass
        apply_live(save=False)
        # don't rewrite the INI just for opening the dialog
        preview.set_state("thinking")

    dlg._select_model = load_model
    model_var.trace_add("write", lambda *_: load_model(model_var.get()))

    # shape buttons
    for i, name in enumerate(shapes.SHAPES):
        cell = tk.Frame(grid, bg=C["surface"], highlightthickness=2,
                        highlightbackground=C["border"], cursor="hand2")
        cell.grid(row=i // 3, column=i % 3, padx=4, pady=4)
        pl = IconPlayer(app, cell, size=40, hole_key="surface")
        pl.set_appearance({"shape": name, "fill": C.get("icon_fill", shapes.PURPLE),
                           "outline": C.get("icon_outline", shapes.ORANGE),
                           "face": True, "size": 32, "motion": "bob",
                           "speed": 1.0}, None, True, "default")
        pl.set_state("idle")
        pl.canvas.pack(padx=6, pady=4)
        tk.Label(cell, text=name, bg=C["surface"], fg=C["fg2"],
                 font=app.f_tiny).pack(pady=(0, 4))
        def pick_shape(ev=None, n=name):
            shape_var.set(n)
            src_var.set("shape")
            icon_var.set("")
            apply_live()
        cell.bind("<Button-1>", pick_shape)
        pl.canvas.bind("<Button-1>", pick_shape)
        shape_btns[name] = cell
        shape_previews[name] = cell

    # colour rows
    def color_row(parent, label, var):
        row = tk.Frame(parent, bg=C["bg"])
        row.pack(fill="x", pady=2)
        tk.Label(row, text=label, bg=C["bg"], fg=C["fg2"],
                 font=app.f_small, width=8, anchor="w").pack(side="left")
        sw = tk.Frame(row, width=18, height=18, bg=var.get() or C["accent"],
                      highlightthickness=1, highlightbackground=C["border"])
        sw.pack(side="left", padx=(0, 6))
        e = tk.Entry(row, textvariable=var, bg=C["surface"], fg=C["fg"],
                     insertbackground=C["fg"], width=10, relief="flat",
                     font=app.f_small)
        e.pack(side="left")

        def on_change(*_):
            c = shapes.parse_color(var.get(), None)
            if c:
                sw.configure(bg=c)
                apply_live()
        var.trace_add("write", lambda *_: on_change())
        # theme swatches
        for key, lab in (("accent", ""), ("accent2", ""),
                         ("icon_fill", ""), ("icon_outline", "")):
            col = C.get(key) or C["accent"]
            chip = tk.Frame(row, bg=col, width=14, height=14,
                            highlightthickness=1,
                            highlightbackground=C["border"], cursor="hand2")
            chip.pack(side="left", padx=2)
            chip.bind("<Button-1>", lambda ev, c=col, v=var: v.set(c))
        return row

    color_row(ctrls, "Fill", fill_var)
    color_row(ctrls, "Outline", out_var)

    opt = tk.Frame(ctrls, bg=C["bg"])
    opt.pack(fill="x", pady=4)
    tk.Checkbutton(opt, text="Face", variable=face_var, bg=C["bg"],
                   fg=C["fg"], selectcolor=C["surface"],
                   activebackground=C["bg"], font=app.f_small,
                   highlightthickness=0,
                   command=lambda: apply_live()).pack(side="left")
    tk.Label(opt, text="Size", bg=C["bg"], fg=C["fg2"],
             font=app.f_small).pack(side="left", padx=(12, 4))
    for sz in (32, 64):
        tk.Radiobutton(opt, text=str(sz), variable=size_var, value=sz,
                       bg=C["bg"], fg=C["fg"], selectcolor=C["surface"],
                       activebackground=C["bg"], font=app.f_small,
                       highlightthickness=0,
                       command=lambda: apply_live()).pack(side="left")

    mot = tk.Frame(ctrls, bg=C["bg"])
    mot.pack(fill="x", pady=4)
    tk.Label(mot, text="Motion", bg=C["bg"], fg=C["fg2"],
             font=app.f_small).pack(side="left")
    for name in shapes.MOTIONS:
        tk.Radiobutton(mot, text=name, variable=motion_var, value=name,
                       bg=C["bg"], fg=C["fg"], selectcolor=C["surface"],
                       activebackground=C["bg"], font=app.f_tiny,
                       highlightthickness=0,
                       command=lambda: apply_live()).pack(side="left")

    spd = tk.Frame(ctrls, bg=C["bg"])
    spd.pack(fill="x", pady=4)
    tk.Label(spd, text="Speed", bg=C["bg"], fg=C["fg2"],
             font=app.f_small).pack(side="left")
    scale = tk.Scale(spd, from_=0.25, to=3.0, resolution=0.05,
                     orient="horizontal", variable=speed_var,
                     bg=C["bg"], fg=C["fg"], troughcolor=C["surface"],
                     highlightthickness=0, length=180, showvalue=True,
                     command=lambda v: apply_live())
    scale.pack(side="left", padx=8)

    def on_src(*_):
        if src_var.get() == "shape":
            icon_var.set("")
        apply_live()
    src_var.trace_add("write", lambda *_: on_src())

    def pick_pack(ev=None):
        sel = pack_list.curselection()
        if not sel:
            return
        ident = pack_ids[sel[0]]
        icon_var.set(ident)
        src_var.set("pack")
        apply_live()
    pack_list.bind("<Double-Button-1>", pick_pack)
    pack_list.bind("<Return>", pick_pack)

    def browse():
        from tkinter import filedialog
        path = filedialog.askdirectory(
            parent=dlg, title="Choose an icon pack folder (with manifest.json "
            "or strip-32.png)")
        if path:
            icon_var.set(path)
            src_var.set("pack")
            apply_live()

    def clear_pack():
        icon_var.set("")
        src_var.set("shape")
        pack_list.selection_clear(0, "end")
        apply_live()

    flatbtn(pack_btns, "Use selected", pick_pack, C["raised"], C["fg2"],
            C["select"], app.f_tiny, padx=8, pady=3).pack(side="left", padx=(0, 6))
    flatbtn(pack_btns, "Browse\u2026", browse, C["raised"], C["fg2"],
            C["select"], app.f_tiny, padx=8, pady=3).pack(side="left", padx=(0, 6))
    flatbtn(pack_btns, "Clear pack", clear_pack, C["raised"], C["fg2"],
            C["select"], app.f_tiny, padx=8, pady=3).pack(side="left")

    def on_close():
        if hasattr(dlg, "_unbind_theme_wheel"):
            try:
                dlg._unbind_theme_wheel()
            except tk.TclError:
                pass
        preview.pause()
        preview.destroy()
        for pl in list(shape_previews.values()):
            pass
        # destroy shape-grid players (they're in app._icon_players)
        # The cells go away with the dialog; clean players whose canvas
        # is a descendant of dlg.
        for p in list(app._icon_players):
            try:
                if str(p.canvas.winfo_toplevel()) == str(dlg):
                    p.destroy()
            except tk.TclError:
                p.destroy()
        app._appearance_win = None
        dlg.destroy()

    dlg._close_appearance = on_close
    dlg.protocol("WM_DELETE_WINDOW", on_close)
    foot = tk.Frame(dlg, bg=C["bg"])
    foot.pack(fill="x", padx=16, pady=(0, 12))
    flatbtn(foot, "Close", on_close, C["accent"], C["bg"], C["think_hot"]
            if "think_hot" in C else C["think"], app.f_small_b,
            padx=16, pady=5).pack(side="right")
    tk.Label(foot, text="See ICONS.md and THEMES.md for the drop-in formats.",
             bg=C["bg"], fg=C["faint"], font=app.f_tiny,
             anchor="w").pack(side="left")

    load_model(state["model"])
    show_tab(tab)
    try:
        dlg.grab_set()
    except Exception:
        pass



def open_settings_dialog(app):
    """The Little Nemo engine settings (same fields and code path as
    before: resolve_settings / validate_settings / apply_settings)."""
    root = app.root
    ctx = app.settings_ctx or {}
    argv = ctx.get("argv", [])
    values, found_path = proto.resolve_settings(argv)
    target = (found_path or ctx.get("config_arg")
              or os.path.join(os.getcwd(), "littlenemo.ini"))
    dlg = tk.Toplevel(root)
    dlg.title("Little Nemo engine settings")
    dlg.configure(bg=BG)
    dlg.transient(root)
    form = tk.Frame(dlg, bg=BG)
    form.pack(fill="both", expand=True, padx=14, pady=12)
    r = 0

    def head(text):
        nonlocal r
        tk.Label(form, text=text, bg=BG, fg=FG, font=app.f_small_b,
                 anchor="w").grid(row=r, column=0, columnspan=2, sticky="w",
                                  pady=(8, 2))
        r += 1

    def note(text):
        nonlocal r
        tk.Label(form, text=text, bg=BG, fg=DIM, anchor="w", justify="left",
                 font=app.f_tiny, wraplength=460).grid(
                     row=r, column=0, columnspan=2, sticky="w", pady=1)
        r += 1

    def entry_row(text, initial):
        nonlocal r
        tk.Label(form, text=text, bg=BG, fg=FG2, anchor="w",
                 font=app.f_small).grid(row=r, column=0, sticky="w",
                                        padx=(0, 8), pady=1)
        e = tk.Entry(form, bg=SURFACE, fg=FG, insertbackground=FG, width=50,
                     relief="flat", font=app.f_small)
        e.insert(0, initial)
        e.grid(row=r, column=1, sticky="ew", pady=2)
        r += 1
        return e

    head("Engine")
    tk.Label(form, text="mode", bg=BG, fg=FG2, anchor="w",
             font=app.f_small).grid(row=r, column=0, sticky="w",
                                    padx=(0, 8), pady=1)
    mode_var = tk.StringVar(value=values["mode"] or "max")
    om = tk.OptionMenu(form, mode_var, *proto.MODES)
    om.configure(bg=SURFACE, fg=FG, activebackground=SELECT,
                 highlightthickness=0, relief="flat")
    om.grid(row=r, column=1, sticky="w", pady=1)
    r += 1
    auto_var = tk.BooleanVar(value=values["threads"] in ("", "0"))
    e_threads = entry_row("threads", "" if auto_var.get() else values["threads"])
    tk.Checkbutton(form, text="threads: let the mode decide (auto)",
                   variable=auto_var, bg=BG, fg=FG2, activebackground=BG,
                   selectcolor=SURFACE, font=app.f_small,
                   command=lambda: e_threads.configure(
                       state="disabled" if auto_var.get() else "normal")).grid(
                           row=r, column=1, sticky="w", pady=1)
    r += 1
    if auto_var.get():
        e_threads.configure(state="disabled")
    simd_var = tk.BooleanVar(value=values["simd"])
    tk.Checkbutton(form, text="simd: AVX2 projections where available",
                   variable=simd_var, bg=BG, fg=FG2, activebackground=BG,
                   selectcolor=SURFACE, font=app.f_small).grid(
                       row=r, column=0, columnspan=2, sticky="w", pady=1)
    r += 1
    note("The engine checks the CPU itself and stays scalar on hardware "
         "without AVX2; the status line shows what the engine actually ran.")
    e_max_new = entry_row("max_new", values["max_new"])
    e_max_pos = entry_row("max_pos", values["max_pos"])
    note("Blank max_new / max_pos fall back to the engine defaults "
         "(256 / 4096).")
    head("Paths (local engine)")
    e_engine = entry_row("engine", values["engine"])
    e_ckpt = entry_row("checkpoint", values["checkpoint"])
    e_tok = entry_row("tokenizer", values["tokenizer"])
    note("Relative paths resolve against the settings file's folder.")
    head("Remote engine")
    e_host = entry_row("host", values["host"])
    e_port = entry_row("port", values["port"])
    note("A remote engine keeps its own mode, threads, simd, max_new and "
         "max_pos. With a host set, Apply saves everything but only "
         "reconnects.")
    head("SuperNemo")
    note("SuperNemo's engine is configured in the [supernemo] section of the "
         "settings file (engine = command line, or host / port) or with "
         "--supernemo-engine. Until NTSE has its MoE + vision engine path, "
         "it stays unavailable.")
    note(f"Settings file: {target}"
         + ("" if found_path else " (will be created by Apply)"))
    err = tk.Label(form, text="", bg=BG, fg=ERR, anchor="w", justify="left",
                   font=app.f_small, wraplength=460)
    err.grid(row=r, column=0, columnspan=2, sticky="w", pady=(6, 2))
    r += 1
    btns = tk.Frame(form, bg=BG)
    btns.grid(row=r, column=0, columnspan=2, sticky="e", pady=(4, 0))

    def gather():
        return {"engine": e_engine.get(), "checkpoint": e_ckpt.get(),
                "tokenizer": e_tok.get(), "mode": mode_var.get(),
                "threads": "" if auto_var.get() else e_threads.get(),
                "simd": bool(simd_var.get()), "max_new": e_max_new.get(),
                "max_pos": e_max_pos.get(), "host": e_host.get(),
                "port": e_port.get()}

    def on_apply():
        raw = gather()
        errors = proto.validate_settings(raw)
        if errors:
            err.configure(text="\n".join(f"{k}: {v}" for k, v in errors.items()))
            return
        if app.model == "littlenemo" and (app.busy or not app.hero):
            if not messagebox.askyesno(
                    "Restart engine", "Applying these settings restarts the "
                    "Little Nemo engine and clears the current conversation. "
                    "Continue?", parent=dlg):
                return
        errors, result = proto.apply_settings(raw, target)
        if errors:
            err.configure(text="\n".join(f"{k}: {v}" for k, v in errors.items()))
            return
        dlg.destroy()
        # The applied settings become the session's settings, so a
        # later restart uses them (Archie Round 4 P1-2, 2026-11-25).
        app.targets["littlenemo"] = {"cmd": result["cmd"],
                                     "host": result["host"],
                                     "port": result["port"]}
        if app.model == "littlenemo":
            if app.busy:
                app.on_stop()
                app.busy = False
            app.sys_line("[settings applied; engine restarting, new "
                         "conversation]")
            app.stats.reset_engine()
            app.select_model("littlenemo", announce=False)
        else:
            app._refresh_sidebar()

    flatbtn(btns, "Apply", on_apply, ACCENT, BG, "#d4d4d8", app.f_small_b,
            padx=12, pady=4).pack(side="left", padx=(0, 6))
    flatbtn(btns, "Cancel", dlg.destroy, RAISED, FG2, SELECT, app.f_small,
            padx=12, pady=4).pack(side="left")
    try:
        dlg.grab_set()
    except Exception:
        pass
