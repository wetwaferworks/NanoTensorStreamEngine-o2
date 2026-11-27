"""Voice dictation for NTSE Chat: a big mic button next to the composer.

Click the mic to start, click again to stop (or hold it down and let go for
push-to-talk). While you talk, partial text streams into the composer in a
dim colour and firms up as each phrase is finalized. Nothing is sent: the
text stays in the composer for review, Enter sends as usual.
Shortcut: Ctrl+Shift+Space toggles dictation.

Speech recognition runs in parakeet-c.exe, the native pure-C Parakeet TDT
runtime (no Python, no WSL in the speech path). This module only spawns it
and talks its line protocol over stdin/stdout:
    -> START | STOP | CANCEL | PING | QUIT
    <- READY | REC <mic> | LEVEL <0..1> <speaking> | PARTIAL <seg> <text>
       FINAL <seg> <text> | STAT <seg> k=v... | DONE | ERR <msg>
A reader thread hands lines to the Tk thread through a queue polled by
after(); no locks of our own.
"""
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk

HERE = os.path.dirname(os.path.abspath(__file__))
EXE_NAME = "parakeet-c.exe" if os.name == "nt" else "parakeet-c"

# Overrides from the settings file ([dictation] parakeet_exe, parakeet_model),
# set by configure(). Blank means "search".
_configured = {"exe": "", "model": ""}


def configure(exe="", model="", base=None):
    """Remember the settings-file paths. A relative path resolves against
    base (the settings file's folder)."""
    def resolve(p):
        p = (p or "").strip()
        if not p or os.path.isabs(p) or not base:
            return p
        return os.path.normpath(os.path.join(base, p))
    _configured["exe"] = resolve(exe)
    _configured["model"] = resolve(model)


def exe_candidates():
    """Where the Parakeet runtime is looked for, in order: the settings
    file, PARAKEET_EXE, the built tree this GUI belongs to (parakeet/bin
    beside the repository root), a bin folder beside gui/, a parakeet folder
    beside this file, and a per-user copy in the NemoChat app-data folder."""
    local = os.environ.get("LOCALAPPDATA") or ""
    return [
        _configured["exe"],
        os.environ.get("PARAKEET_EXE", ""),
        os.path.join(HERE, "..", "..", "parakeet", "bin", EXE_NAME),
        os.path.join(HERE, "..", "bin", EXE_NAME),
        os.path.join(HERE, "parakeet", EXE_NAME),
        os.path.join(local, "NemoChat", "parakeet", EXE_NAME) if local else "",
    ]


def _model_dir():
    """The settings file's parakeet_model, else PARAKEET_MODEL, else the
    Windows user environment (so shortcuts started before the variable
    existed still find it), else "" and parakeet-c searches its own default
    locations."""
    if _configured["model"]:
        return _configured["model"]
    v = os.environ.get("PARAKEET_MODEL", "")
    if not v and os.name == "nt":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as k:
                v = winreg.QueryValueEx(k, "PARAKEET_MODEL")[0]
        except OSError:
            v = ""
    return v


REC_RED = "#ef4444"
REC_RED_HOT = "#f87171"


def find_exe():
    """The first Parakeet runtime found, or None."""
    for p in exe_candidates():
        if p and os.path.isfile(p):
            return os.path.normpath(p)
    import shutil
    return shutil.which(EXE_NAME)


NOT_FOUND = "mic: parakeet-c not found (Settings, parakeet_exe)"


class ParakeetProc:
    """parakeet-c.exe --stdio, started on first use and kept warm."""

    def __init__(self, exe, model_dir):
        self.exe, self.model_dir = exe, model_dir
        self.p = None
        self.lines = queue.Queue()
        self.ready = False

    def alive(self):
        return self.p is not None and self.p.poll() is None

    def start(self):
        if self.alive():
            return
        flags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        self.ready = False
        self.p = subprocess.Popen(
            [self.exe, "--stdio"] + (["--model", self.model_dir] if self.model_dir else []),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, creationflags=flags, bufsize=0)
        threading.Thread(target=self._reader, args=(self.p,), daemon=True).start()

    def _reader(self, p):
        for raw in iter(p.stdout.readline, b""):
            self.lines.put(raw.decode("utf-8", "replace").rstrip("\r\n"))
        self.lines.put("EXIT")

    def send(self, cmd):
        if not self.alive():
            return False
        try:
            self.p.stdin.write((cmd + "\n").encode("utf-8"))
            self.p.stdin.flush()
            return True
        except OSError:
            return False

    def close(self):
        if self.alive():
            self.send("QUIT")
            try:
                self.p.wait(timeout=1.5)
            except subprocess.TimeoutExpired:
                self.p.kill()


class Dictation:
    def __init__(self, app, parent):
        self.app = app
        self.root = app.root
        self.text = app.input
        k = getattr(app, "k", 1.0) or 1.0
        self.size = max(40, int(round(44 * k)))   # big target: typing is the slow path
        C = getattr(app, "C", {}) or {}
        self.c_bg = C.get("surface", "#1d1d21")
        self.c_btn = C.get("raised", "#26262b")
        self.c_hover = C.get("select", "#2a2a30")
        self.c_fg = C.get("fg", "#ececf1")
        self.c_dim = C.get("faint", C.get("dim", "#8e8e98"))
        self.c_accent = C.get("accent", "#a970ff")
        self.proc = None
        self.refresh()
        self.state = "idle"        # idle | loading | rec | finishing
        self.finals = []
        self.partial = ""
        self.level = 0.0
        self.t_rec = 0.0
        self.t_press = 0.0
        self.pending_start = False
        self.last_stat = ""

        self.frame = tk.Frame(parent, bg=self.c_bg)
        s = self.size
        self.cv = tk.Canvas(self.frame, width=s, height=s, bg=self.c_bg,
                            highlightthickness=0, bd=0, cursor="hand2")
        self.cv.pack(side="left")
        self.lbl = tk.Label(self.frame, text="Click to dictate", bg=self.c_bg, fg=self.c_fg,
                            font=getattr(app, "f_small_b", None) or getattr(app, "f_small", None),
                            cursor="hand2")
        self.lbl.pack(side="left", padx=(6, 0))
        self.meter = tk.Canvas(self.frame, width=int(60 * k), height=max(6, int(8 * k)),
                               bg=self.c_bg, highlightthickness=0, bd=0)
        self.cv.bind("<ButtonPress-1>", self._press)
        self.cv.bind("<ButtonRelease-1>", self._release)
        self.lbl.bind("<ButtonPress-1>", self._press)
        self.lbl.bind("<ButtonRelease-1>", self._release)
        self.cv.bind("<Enter>", lambda e: self._draw(hover=True))
        self.cv.bind("<Leave>", lambda e: self._draw(hover=False))
        self.root.bind_all("<Control-Shift-space>", lambda e: (self.toggle(), "break")[1])
        self.text.tag_configure("dict_partial", foreground=self.c_dim)
        self._draw()
        self._show_found()
        self.root.after(30, self._poll)
        self.root.bind("<Destroy>", self._on_destroy, add="+")

    # ------------------------------------------------------------ drawing
    def _draw(self, hover=False):
        cv, s = self.cv, self.size
        cv.delete("all")
        rec = self.state in ("rec", "finishing")
        if rec:
            ring = 2 + int(self.level * s * 0.18)
            cv.create_oval(1, 1, s - 1, s - 1, fill=REC_RED_HOT if hover else REC_RED,
                           outline=REC_RED_HOT, width=min(ring, 6))
            q = s * 0.32
            cv.create_rectangle(q, q, s - q, s - q, fill="white", outline="")
            return
        fill = self.c_hover if hover else self.c_btn
        cv.create_oval(1, 1, s - 1, s - 1, fill=fill, outline=self.c_accent if self.state == "loading" else fill, width=2)
        fg = self.c_fg if self.proc else self.c_dim
        cx, w = s / 2, s * 0.10                    # capsule half-width
        top, bot = s * 0.17, s * 0.55
        lw = max(2, int(round(s / 15)))
        cv.create_oval(cx - w, top, cx + w, top + 2 * w, fill=fg, outline="")
        cv.create_rectangle(cx - w, top + w, cx + w, bot - w, fill=fg, outline="")
        cv.create_oval(cx - w, bot - 2 * w, cx + w, bot, fill=fg, outline="")
        a = s * 0.19                               # cradle
        cv.create_arc(cx - a, s * 0.30, cx + a, s * 0.66, start=200, extent=140,
                      style="arc", outline=fg, width=lw)
        cv.create_line(cx, s * 0.66, cx, s * 0.78, fill=fg, width=lw)
        cv.create_line(cx - s * 0.12, s * 0.79, cx + s * 0.12, s * 0.79, fill=fg, width=lw)

    on_label = None  # set by the chat window to refit the composer row

    def _say(self, **kw):
        """Change the status label, then let the composer refit its row
        (a longer message may need the hint's room)."""
        self.lbl.configure(**kw)
        if self.on_label is not None:
            try:
                self.on_label()
            except Exception:
                pass

    def _draw_meter(self):
        m = self.meter
        m.delete("all")
        w, h = int(m["width"]), int(m["height"])
        m.create_rectangle(0, 0, w, h, fill=self.c_btn, outline="")
        m.create_rectangle(0, 0, int(w * self.level), h, fill=REC_RED if self.level < 0.85 else "#fbbf24", outline="")

    # ------------------------------------------------------------ input
    def _press(self, e):
        self.t_press = time.monotonic()
        if self.state in ("idle",):
            self.start()
            self._ptt = True
        else:
            self._ptt = False
            self.stop()

    def _release(self, e):
        held = time.monotonic() - self.t_press
        if getattr(self, "_ptt", False) and held > 0.6 and self.state in ("rec", "loading"):
            self.stop()          # push-to-talk: let go to finish
        self._ptt = False

    def refresh(self):
        """Look for the runtime again (at start-up, after the settings
        change, and on a click when none was found before)."""
        exe = find_exe()
        model = _model_dir()
        if self.proc and self.proc.exe == exe and self.proc.model_dir == model:
            return
        if self.proc:
            self.proc.close()
        self.proc = ParakeetProc(exe, model) if exe else None
        self._show_found()

    def _show_found(self):
        lbl = getattr(self, "lbl", None)  # not built yet during __init__
        if lbl is None:
            return
        if not self.proc:
            lbl.configure(text=NOT_FOUND)
        elif lbl.cget("text") == NOT_FOUND:
            lbl.configure(text="")

    def toggle(self):
        if self.state == "idle":
            self.start()
        else:
            self.stop()

    def start(self, sim=None):
        """Begin dictation. sim=<wav path> replays a WAV instead of the mic (tests)."""
        self._start_cmd = f"SIMFILE {sim}" if sim else "START"
        if not self.proc:
            self.refresh()  # the runtime may have been built or set since
        if not self.proc:
            self._say(text=NOT_FOUND, fg=REC_RED)
            return
        # anchor the dictation span at the cursor
        t = self.text
        try:
            idx = t.index("insert")
        except tk.TclError:
            idx = "end-1c"
        before = t.get(f"{idx} -1c", idx)
        self.lead = "" if (not before or before.isspace()) else " "
        t.mark_set("dict_a", idx)
        t.mark_gravity("dict_a", "left")
        t.mark_set("dict_b", idx)
        t.mark_gravity("dict_b", "right")
        self.finals, self.partial = [], ""
        self.t_rec = time.monotonic()
        if not self.proc.alive():
            self.proc.start()
        self.warn_text = ""
        if self.proc.ready:
            self.proc.send(self._start_cmd)
            self._set_state("rec")
        else:
            self.pending_start = True
            self._set_state("loading")

    def stop(self):
        if self.state == "loading":
            self.pending_start = False
            self._set_state("idle")
            return
        if self.state == "rec":
            self.proc.send("STOP")
            self._set_state("finishing")

    def _set_state(self, st):
        self.state = st
        if st == "rec":
            self._say(text="Listening\u2026 0:00", fg=REC_RED)
            self.meter.pack(side="left", padx=(6, 0))
        elif st == "loading":
            self._say(text="Starting mic\u2026", fg=self.c_dim)
        elif st == "finishing":
            self._say(text="Finishing\u2026", fg=self.c_dim)
        else:
            self.meter.pack_forget()
            tail = f"  \u00b7  {self.last_stat}" if self.last_stat else "  \u00b7  Ctrl+Shift+Space"
            self._say(text="Click to dictate" + tail, fg=self.c_fg)
        self._draw()

    # ------------------------------------------------------------ composer
    def _render(self):
        t = self.text
        body = " ".join(x for x in self.finals if x)
        try:
            t.delete("dict_a", "dict_b")
            pos = "dict_a"
            if body or self.partial:
                t.insert(pos, self.lead)
            if body:
                t.insert("dict_b", body)
            if self.partial:
                t.insert("dict_b", (" " if body else "") + self.partial, ("dict_partial",))
            t.see("dict_b")
            t.mark_set("insert", "dict_b")
        except tk.TclError:
            return
        if hasattr(self.app, "_input_changed"):
            try:
                self.app._input_changed()
            except Exception:
                pass

    # ------------------------------------------------------------ protocol
    def _poll(self):
        try:
            for _ in range(200):
                line = self.proc.lines.get_nowait() if self.proc else None
                if line is None:
                    break
                self._handle(line)
        except queue.Empty:
            pass
        if self.state == "rec" and not getattr(self, "warn_text", ""):
            secs = int(time.monotonic() - self.t_rec)
            self._say(text=f"Listening\u2026 {secs // 60}:{secs % 60:02d}")
        try:
            self.root.after(30, self._poll)
        except tk.TclError:
            pass

    def _handle(self, line):
        kind, _, rest = line.partition(" ")
        if kind == "READY":
            self.proc.ready = True
            if self.pending_start:
                self.pending_start = False
                self.proc.send(getattr(self, "_start_cmd", "START"))
                self._set_state("rec")
        elif kind == "LEVEL":
            try:
                self.level = float(rest.split()[0])
            except (ValueError, IndexError):
                self.level = 0.0
            self._draw()
            self._draw_meter()
        elif kind in ("PARTIAL", "FINAL"):
            _, _, text = rest.partition(" ")
            if kind == "PARTIAL":
                self.partial = text
            else:
                self.finals.append(text)
                self.partial = ""
            self._render()
        elif kind == "STAT":
            parts = dict(p.split("=", 1) for p in rest.split()[1:] if "=" in p)
            try:
                self.last_stat = f"last phrase {float(parts.get('total', 0)):.2f}s for {float(parts.get('audio', 0)):.1f}s of speech"
            except ValueError:
                pass
        elif kind == "DONE":
            self.partial = ""
            self._render()
            try:
                self.text.tag_remove("dict_partial", "1.0", "end")
                self.text.focus_set()
            except tk.TclError:
                pass
            self._set_state("idle")
        elif kind == "WARN":
            self._say(text="mic silent? check NVIDIA Broadcast / PARAKEET_MIC", fg="#fbbf24")
            self.warn_text = rest
        elif kind == "ERR":
            self._say(text="mic: " + rest[:60], fg=REC_RED)
            if self.state != "idle":
                self.state = "idle"
                self.meter.pack_forget()
                self._draw()
        elif kind == "EXIT":
            self.proc.ready = False
            if self.state != "idle":
                self.state = "idle"
                self.meter.pack_forget()
                self._draw()
                self._say(text="mic: speech runtime exited", fg=REC_RED)

    def _on_destroy(self, e):
        if e.widget is self.root and self.proc:
            self.proc.close()


def attach(app, tools_frame, side="left"):
    """Add the dictation button to the composer tool row; returns Dictation."""
    d = Dictation(app, tools_frame)
    d.frame.pack(side=side, padx=(8, 4))
    return d


def prewarm(d):
    """Start parakeet-c in the background so the first click is instant."""
    if d.proc and not d.proc.alive():
        d.proc.start()
