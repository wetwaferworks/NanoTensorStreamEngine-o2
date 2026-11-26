#!/usr/bin/env python3
"""nemo_chat.py - Little Nemo chat front end.

The ONLY Python piece of Little Nemo, by JD's standing rule: the front
end is a Python GUI that calls the pure-C backend. This file renders the
conversation, the model's thinking, and the telemetry deck, and speaks
PROTOCOL.md v2 to the engine over a subprocess pipe or a TCP socket
(so the same window works against an engine on your own machine or one
running on Elfie's computer).

    python3 nemo_chat.py [--engine "../nemo --serve-stdio"] [--host H --port P]

Configuration can also come from an INI file, so the window can be
launched by clicking with no arguments at all. The GUI looks for
littlenemo.ini in the current directory, beside this script, and in
the script's parent directory (or at the path given by --config):

    [paths]
    engine     = ../bin/nemo-engine.exe
    checkpoint = D:/models/Nemotron-Nano-9B-v2
    tokenizer  = D:/models/Nemotron-Nano-9B-v2/tokenizer.json

    [engine]
    mode    = max          ; minimum | medium | max | ultra
    threads = 0            ; 0 = the mode decides
    simd    = false        ; true = AVX2 projections where available
    max_new = 256
    max_pos = 4096

    [remote]
    host =                   ; set these instead of [paths] to talk
    port =                   ; to an engine on another machine

Relative paths in the INI resolve against the INI file's own
directory. Command line arguments override the INI.

The chat window also carries a Settings menu (top right): the
same configuration, editable at runtime. Applying validates
every field, writes the values back to the INI file (unrelated
sections and keys are preserved), and restarts the engine with
the new configuration; the READY line the engine answers with
is shown in the status bar as the receipt, next to the core
count this machine reports. The settings code path
(resolve_settings / validate_settings / apply_settings) has no
Tk dependency and is covered headlessly by
tests/test_gui_settings.py.

The client half (NemoClient) has no Tk dependency and is unit-testable
headlessly; see tests/test_gui_client.py.
"""

import configparser
import os
import re
import subprocess
import sys
import threading

# Layer types for NVIDIA-Nemotron-Nano-9B-v2, from the checkpoint's
# hybrid_override_pattern (M = mamba, - = mlp, * = attention).
LAYER_PATTERN = "M-M-M-MM-M-M-M*-M-M-M*-M-M-M-M*-M-M-M-M*-M-MM-M-M-M-M-M-"
LAYER_TYPES = [
    {"M": "mamba", "-": "mlp", "*": "attention"}[c] for c in LAYER_PATTERN
]


def escape(text):
    return text.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def unescape(text):
    out = []
    i = 0
    while i < len(text):
        c = text[i]
        if c == "\\" and i + 1 < len(text):
            n = text[i + 1]
            out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(n, n))
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def split_cmdline_windows(s):
    """Split a command line by the Windows (CommandLineToArgvW /
    Microsoft C runtime) rules, as a pure function so it can be
    unit-tested on any platform: whitespace delimits arguments
    outside quotes; a double quote toggles the in-quote state
    and is consumed; backslashes are literal except before a
    double quote, where 2n backslashes mean n literal ones and
    the quote still delimits, and 2n+1 mean n literal ones and
    a literal quote. (Sol M5: POSIX shlex destroyed the
    backslashes in Windows paths in --engine overrides.)"""
    args = []
    cur = []
    in_quotes = False
    have_arg = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            j = i
            while j < n and s[j] == "\\":
                j += 1
            run = j - i
            if j < n and s[j] == '"':
                cur.append("\\" * (run // 2))
                if run % 2 == 1:
                    cur.append('"')
                else:
                    in_quotes = not in_quotes
                have_arg = True
                i = j + 1
            else:
                cur.append("\\" * run)
                have_arg = True
                i = j
        elif c == '"':
            in_quotes = not in_quotes
            have_arg = True
            i += 1
        elif c in " \t" and not in_quotes:
            if have_arg:
                args.append("".join(cur))
                cur = []
                have_arg = False
            i += 1
        else:
            cur.append(c)
            have_arg = True
            i += 1
    if have_arg:
        args.append("".join(cur))
    return args


def _kv(tokens):
    d = {}
    for tok in tokens:
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = v
    return d


def parse_line(line):
    """One protocol line -> (kind, payload). Kinds: ready, think, tok,
    layer, tick, stats, done, error. Unknown lines are ('raw', line)."""
    line = line.rstrip("\n")
    if line.startswith("READY "):
        return ("ready", _kv(line.split()[2:]))
    if line.startswith("THINK "):
        return ("think", unescape(line[6:]))
    if line.startswith("TOK "):
        return ("tok", unescape(line[4:]))
    if line.startswith("LAYER "):
        parts = line.split()
        return ("layer", {"idx": int(parts[1]), "type": parts[2]})
    if line.startswith("TICK "):
        d = _kv(line.split()[1:])
        for k in ("layer", "next", "active", "staged", "tok"):
            if k in d:
                d[k] = int(d[k])
        for k in ("ms_tok", "gbs"):
            if k in d:
                d[k] = float(d[k])
        return ("tick", d)
    if line.startswith("STATS "):
        d = _kv(line.split()[1:])
        for k in ("tokens",):
            if k in d:
                d[k] = int(d[k])
        for k in ("ms", "tps", "ms_per_tok", "gbs_avg", "peak_mb"):
            if k in d:
                d[k] = float(d[k])
        return ("stats", d)
    if line == "DONE":
        return ("done", None)
    if line.startswith("ERROR "):
        return ("error", line[6:])
    return ("raw", line)


class NemoClient:
    """Speaks the protocol to one engine. on_event(kind, payload) is
    called from the reader thread; the GUI marshals to its own thread."""

    def __init__(self, on_event):
        self.on_event = on_event
        self._w = None
        self._proc = None
        self._sock = None
        self._lock = threading.Lock()

    def connect_process(self, cmd):
        # Structured argument list, never a shell: configured
        # paths from the INI file must not be able to smuggle
        # shell metacharacters into the command that runs
        # (Codex Astra review M-I2, 2026-11-25). A string is
        # still accepted for the explicit --engine CLI override
        # and is split, not executed by a shell: with POSIX
        # shlex on POSIX, and with Windows command-line rules
        # on Windows (ChatGPT Sol review M5, 2026-11-25:
        # shlex.split eats the backslashes in a Windows path
        # like C:\\engines\\nemo, mangling the override before
        # it ever runs).
        if isinstance(cmd, str):
            if os.name == "nt":
                cmd = split_cmdline_windows(cmd)
            else:
                import shlex

                cmd = shlex.split(cmd)
        if os.name == "nt" and cmd and str(cmd[0]).lower().endswith(".py"):
            # Windows CreateProcess cannot execute a .py file
            # directly (WinError 193, "%1 is not a valid Win32
            # application"); the shebang that carries it on
            # POSIX means nothing there. Run the script with
            # this interpreter instead (Archie Round 4 P1-1,
            # 2026-11-25).
            cmd = [sys.executable] + list(cmd)
        self._proc = subprocess.Popen(
            cmd, shell=False, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._w = self._proc.stdin
        self._start_reader(self._proc.stdout)

    def connect_tcp(self, host, port):
        import socket

        self._sock = socket.create_connection((host, port), timeout=10)
        # The 10 s timeout was for connection establishment
        # only; left in place it governed every read and killed
        # healthy idle connections (Codex Astra review M-I1,
        # 2026-11-25). Reads now block until data or close.
        self._sock.settimeout(None)
        f = self._sock.makefile("rw", encoding="utf-8", newline="\n")
        self._w = f
        self._start_reader(f)

    def _start_reader(self, r):
        def run():
            try:
                for line in r:
                    kind, payload = parse_line(line)
                    self.on_event(kind, payload)
            except Exception as exc:  # connection broke
                self.on_event("error", str(exc))
            self.on_event("closed", None)

        threading.Thread(target=run, daemon=True).start()

    def _send(self, line):
        with self._lock:
            if self._w is None:
                raise RuntimeError("not connected")
            self._w.write(line + "\n")
            self._w.flush()

    def send_prompt(self, text):
        self._send("PROMPT " + escape(text))

    def stop(self):
        self._send("STOP")

    def reset(self):
        self._send("RESET")

    def quit(self):
        try:
            self._send("QUIT")
        except Exception:
            pass


# ---------------------------------------------------------------- GUI ---

BG = "#16161e"
PANEL = "#1f2335"
FG = "#c0caf5"
DIM = "#565f89"
ACCENT = "#7aa2f7"
THINK_COLOR = "#bb9af7"
ANSWER_COLOR = "#9ece6a"
LAYER_COLORS = {"mamba": "#7aa2f7", "mlp": "#3b4261", "attention": "#ff9e64"}
LAYER_LIT = {"mamba": "#a9c1ff", "mlp": "#8a94b8", "attention": "#ffc79e"}


def run_gui(engine_cmd=None, host=None, port=None, settings_ctx=None):
    import queue
    import time
    import tkinter as tk
    from tkinter import messagebox
    from tkinter.scrolledtext import ScrolledText

    events = queue.Queue()
    # The live engine connection, with a generation counter: a
    # respawn (settings apply, restart) retires the old client,
    # and events tagged with a retired generation are ignored,
    # so a dying engine cannot mark its replacement disconnected.
    conn = {"client": None, "gen": 0}

    root = tk.Tk()
    root.title("Little Nemo")
    root.configure(bg=BG)
    root.geometry("1060x760")

    state = {
        "mode": "idle", "layer": None, "next": None, "active": 0,
        "staged": 0, "tok": 0, "ms_tok": 0.0, "gbs": 0.0,
        "turn_start": None, "busy": False, "model": "...",
        "turns": 0, "ready": {},
    }

    top = tk.Frame(root, bg=BG)
    top.pack(fill="x", padx=10, pady=(10, 2))
    tk.Label(top, text="Little Nemo", font=("TkDefaultFont", 15, "bold"),
             bg=BG, fg=FG).pack(side="left")
    status = tk.Label(top, text="connecting...", bg=BG, fg=DIM)
    status.pack(side="left", padx=10)
    lamp = tk.Label(top, text="● IDLE", bg=BG, fg=DIM,
                    font=("TkDefaultFont", 11, "bold"))
    lamp.pack(side="right")

    settings_btn = tk.Menubutton(top, text="Settings", bg=PANEL, fg=FG,
                                  activebackground=PANEL, relief="flat")
    settings_menu = tk.Menu(settings_btn, tearoff=0, bg=PANEL, fg=FG,
                            activebackground=ACCENT)
    settings_menu.add_command(label="Engine settings...",
                              command=lambda: open_settings())
    settings_menu.add_command(label="Restart engine",
                              command=lambda: restart_engine())
    settings_btn.configure(menu=settings_menu)
    settings_btn.pack(side="right", padx=(0, 8))

    panes = tk.Frame(root, bg=BG)
    panes.pack(fill="both", expand=True, padx=10, pady=4)
    chat = ScrolledText(panes, bg=PANEL, fg=FG, insertbackground=FG,
                        font=("TkDefaultFont", 11), wrap="word",
                        state="disabled", width=64)
    chat.pack(side="left", fill="both", expand=True)
    chat.tag_config("user", foreground=ACCENT, font=("TkDefaultFont", 11, "bold"))
    chat.tag_config("nemo", foreground=FG)
    chat.tag_config("sys", foreground=DIM)

    right = tk.Frame(panes, bg=BG)
    right.pack(side="right", fill="both", padx=(8, 0))
    tk.Label(right, text="Thinking", bg=BG, fg=THINK_COLOR,
             font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
    think = ScrolledText(right, bg=PANEL, fg=THINK_COLOR,
                         font=("TkDefaultFont", 10, "italic"), wrap="word",
                         state="disabled", width=38)
    think.pack(fill="both", expand=True)

    deck = tk.Frame(root, bg=PANEL)
    deck.pack(fill="x", padx=10, pady=4)
    strip = tk.Canvas(deck, bg=PANEL, height=26, highlightthickness=0)
    strip.pack(fill="x", padx=8, pady=(6, 0))
    cells = []
    readout = tk.Label(deck, bg=PANEL, fg=FG, font=("TkFixedFont", 10),
                       anchor="w", justify="left")
    readout.pack(fill="x", padx=8, pady=(2, 6))

    bottom = tk.Frame(root, bg=BG)
    bottom.pack(fill="x", padx=10, pady=(2, 10))
    entry = tk.Entry(bottom, bg=PANEL, fg=FG, insertbackground=FG,
                     font=("TkDefaultFont", 11))
    entry.pack(side="left", fill="x", expand=True)

    def draw_strip():
        strip.delete("all")
        w = max(strip.winfo_width(), 200)
        n = len(LAYER_TYPES)
        cw = w / n
        for i, t in enumerate(LAYER_TYPES):
            lit = state["layer"] is not None and i <= state["layer"]
            staged = (state["layer"] is not None
                      and state["layer"] < i <= state["layer"] + state["staged"])
            fill = LAYER_LIT[t] if lit else LAYER_COLORS[t]
            outline = "#e0e6ff" if i == state["layer"] else ""
            y0 = 3 if not staged else 8
            strip.create_rectangle(i * cw + 1, y0, (i + 1) * cw - 1, 21,
                                   fill=fill, outline=outline, width=2 if outline else 1)
            if staged:
                strip.create_line(i * cw + 2, 24, (i + 1) * cw - 2, 24,
                                  fill=ACCENT, width=2)
    strip.bind("<Configure>", lambda e: draw_strip())

    def append(widget, text, tag=None):
        widget.configure(state="normal")
        widget.insert("end", text, tag)
        widget.see("end")
        widget.configure(state="disabled")

    def fmt_readout():
        lyr = state["layer"]
        cur = (f"layer {lyr}/{len(LAYER_TYPES) - 1} ({LAYER_TYPES[lyr]})"
               if lyr is not None and 0 <= lyr < len(LAYER_TYPES)
               else "layer -")
        nxt = state["next"]
        nxt_s = (f"next {nxt} ({LAYER_TYPES[nxt]})"
                 if nxt is not None and 0 <= nxt < len(LAYER_TYPES) else "next -")
        el = ""
        if state["turn_start"] is not None:
            el = f"elapsed {time.time() - state['turn_start']:.1f}s"
        tps = (1000.0 / state["ms_tok"]) if state["ms_tok"] > 0 else 0.0
        return (f"{cur}   {nxt_s}   active {state['active']}   "
                f"predict-ahead {state['staged']}   tok {state['tok']}   "
                f"{tps:.1f} tok/s   {state['ms_tok']:.0f} ms/tok   "
                f"{state['gbs']:.1f} GB/s est   {el}")

    def set_mode(mode):
        state["mode"] = mode
        lamp.configure(
            text={"thinking": "● THINKING", "answering": "● ANSWERING",
                  "idle": "● IDLE"}[mode],
            fg={"thinking": THINK_COLOR, "answering": ANSWER_COLOR,
                "idle": DIM}[mode])

    def on_send():
        text = entry.get().strip()
        if not text or state["busy"]:
            return
        entry.delete(0, "end")
        append(chat, "You: ", "user")
        append(chat, text + "\n", "user")
        state["busy"] = True
        state["turn_start"] = time.time()
        state["tok"] = 0
        set_mode("thinking")
        try:
            conn["client"].send_prompt(text)
            state["turns"] += 1
        except Exception as exc:
            append(chat, f"[not connected: {exc}]\n", "sys")
            state["busy"] = False
            set_mode("idle")

    def on_stop():
        try:
            conn["client"].stop()
        except Exception:
            pass

    def on_reset():
        try:
            conn["client"].reset()
        except Exception:
            pass
        state["turns"] = 0
        append(chat, "[conversation reset]\n", "sys")
        think.configure(state="normal")
        think.delete("1.0", "end")
        think.configure(state="disabled")

    tk.Button(bottom, text="Send", command=on_send, bg=ACCENT, fg="#16161e",
              font=("TkDefaultFont", 10, "bold")).pack(side="left", padx=(8, 0))
    tk.Button(bottom, text="Stop", command=on_stop, bg=PANEL, fg=FG).pack(
        side="left", padx=(6, 0))
    tk.Button(bottom, text="Reset", command=on_reset, bg=PANEL, fg=FG).pack(
        side="left", padx=(6, 0))
    entry.bind("<Return>", lambda e: on_send())
    entry.focus_set()

    def poll():
        try:
            while True:
                gen, kind, payload = events.get_nowait()
                if gen != conn["gen"]:
                    continue
                if kind == "ready":
                    state["ready"] = payload
                    state["model"] = payload.get("model", "?")
                    extra = ""
                    if "cores" in payload:
                        extra = (f" · cores {payload.get('cores', '?')}"
                                 f" · threads {payload.get('threads', '?')}"
                                 f" · mode {payload.get('mode', '?')}"
                                 f" · simd "
                                 f"{'on' if payload.get('simd') == '1' else 'off'}")
                    status.configure(
                        text=f"connected · {state['model']} · "
                             f"weights {payload.get('weights', '?')}{extra}")
                elif kind == "think":
                    append(think, payload)
                elif kind == "tok":
                    if state["mode"] != "answering":
                        set_mode("answering")
                        append(chat, "Nemo: ", "nemo")
                    append(chat, payload, "nemo")
                elif kind == "layer":
                    state["layer"] = payload["idx"]
                    draw_strip()
                elif kind == "tick":
                    for k in ("layer", "next", "active", "staged", "tok",
                              "ms_tok", "gbs"):
                        if k in payload:
                            state[k] = payload[k]
                    set_mode(payload.get("mode", state["mode"]))
                    draw_strip()
                elif kind == "stats":
                    append(chat,
                           f"\n[{payload.get('tokens', 0)} tokens · "
                           f"{payload.get('tps', 0):.1f} tok/s · "
                           f"{payload.get('ms_per_tok', 0):.0f} ms/tok · "
                           f"{payload.get('gbs_avg', 0):.1f} GB/s avg est · "
                           f"peak {payload.get('peak_mb', 0):.0f} MB]\n", "sys")
                elif kind == "done":
                    append(chat, "\n")
                    state["busy"] = False
                    state["turn_start"] = None
                    set_mode("idle")
                elif kind == "error":
                    append(chat, f"[error: {payload}]\n", "sys")
                    state["busy"] = False
                    set_mode("idle")
                elif kind == "closed":
                    status.configure(text="disconnected")
        except queue.Empty:
            pass
        readout.configure(text=fmt_readout())
        root.after(120, poll)

    def connect(cmd, h, p):
        """(Re)spawn the engine with an argument list or a remote
        target, retiring any live client. This is the runtime
        refresh the Settings menu applies through: the engine
        holds the conversation state, so a respawn starts a
        fresh session, and the new READY line in the status bar
        is the receipt that the settings were picked up."""
        old = conn["client"]
        if old is not None:
            old.quit()
            try:
                if old._proc is not None:
                    old._proc.terminate()
            except Exception:
                pass
            try:
                if old._sock is not None:
                    old._sock.close()
            except Exception:
                pass
        conn["gen"] += 1
        gen = conn["gen"]
        c = NemoClient(lambda k, pl, g=gen: events.put((g, k, pl)))
        conn["client"] = c
        state["busy"] = False
        state["turns"] = 0
        set_mode("idle")
        if h:
            try:
                c.connect_tcp(h, int(p))
                status.configure(text="connecting...")
            except Exception as exc:
                status.configure(text=f"connect failed: {exc}")
        elif cmd:
            try:
                c.connect_process(cmd)
                status.configure(text="connecting...")
            except Exception as exc:
                status.configure(text=f"engine failed to start: {exc}")
        else:
            status.configure(
                text="no engine configured (see littlenemo.ini or --engine / --host)")

    def restart_engine():
        append(chat, "[engine restarting with the current settings]\n",
               "sys")
        think.configure(state="normal")
        think.delete("1.0", "end")
        think.configure(state="disabled")
        connect(engine_cmd, host, port)

    def open_settings():
        ctx = settings_ctx or {}
        argv = ctx.get("argv", [])
        values, found_path = resolve_settings(argv)
        target = (found_path or ctx.get("config_arg")
                  or os.path.join(os.getcwd(), "littlenemo.ini"))

        dlg = tk.Toplevel(root)
        dlg.title("Little Nemo settings")
        dlg.configure(bg=BG)
        dlg.transient(root)
        form = tk.Frame(dlg, bg=BG)
        form.pack(fill="both", expand=True, padx=12, pady=10)
        r = 0

        def head(text):
            nonlocal r
            tk.Label(form, text=text, bg=BG, fg=ACCENT,
                     font=("TkDefaultFont", 10, "bold"),
                     anchor="w").grid(row=r, column=0, columnspan=2,
                                      sticky="w", pady=(8, 2))
            r += 1

        def note(text):
            nonlocal r
            tk.Label(form, text=text, bg=BG, fg=DIM, anchor="w",
                     justify="left", wraplength=430).grid(
                         row=r, column=0, columnspan=2, sticky="w",
                         pady=1)
            r += 1

        def entry_row(text, initial):
            nonlocal r
            tk.Label(form, text=text, bg=BG, fg=FG, anchor="w").grid(
                row=r, column=0, sticky="w", padx=(0, 8), pady=1)
            e = tk.Entry(form, bg=PANEL, fg=FG, insertbackground=FG,
                         width=46)
            e.insert(0, initial)
            e.grid(row=r, column=1, sticky="ew", pady=1)
            r += 1
            return e

        head("Engine")
        tk.Label(form, text="mode", bg=BG, fg=FG, anchor="w").grid(
            row=r, column=0, sticky="w", padx=(0, 8), pady=1)
        mode_var = tk.StringVar(value=values["mode"] or "max")
        tk.OptionMenu(form, mode_var, *MODES).grid(
            row=r, column=1, sticky="w", pady=1)
        r += 1

        auto_var = tk.BooleanVar(value=values["threads"] in ("", "0"))
        e_threads = entry_row(
            "threads", "" if auto_var.get() else values["threads"])
        tk.Checkbutton(
            form, text="threads: let the mode decide (auto)",
            variable=auto_var, bg=BG, fg=FG, activebackground=BG,
            selectcolor=PANEL,
            command=lambda: e_threads.configure(
                state="disabled" if auto_var.get() else "normal")).grid(
                    row=r, column=1, sticky="w", pady=1)
        r += 1
        if auto_var.get():
            e_threads.configure(state="disabled")

        simd_var = tk.BooleanVar(value=values["simd"])
        tk.Checkbutton(
            form, text="simd: AVX2 projections where available",
            variable=simd_var, bg=BG, fg=FG, activebackground=BG,
            selectcolor=PANEL).grid(row=r, column=0, columnspan=2,
                                     sticky="w", pady=1)
        r += 1
        note("The engine checks the CPU itself and stays scalar on "
             "hardware without AVX2; the status bar shows what the "
             "engine actually ran.")
        e_max_new = entry_row("max_new", values["max_new"])
        e_max_pos = entry_row("max_pos", values["max_pos"])
        note("Blank max_new / max_pos fall back to the engine "
             "defaults (256 / 4096).")

        head("Paths (local engine)")
        e_engine = entry_row("engine", values["engine"])
        e_ckpt = entry_row("checkpoint", values["checkpoint"])
        e_tok = entry_row("tokenizer", values["tokenizer"])
        note("Relative paths resolve against the settings file's "
             "folder.")

        head("Remote engine")
        e_host = entry_row("host", values["host"])
        e_port = entry_row("port", values["port"])
        note("A remote engine keeps its own mode, threads, simd, "
             "max_new and max_pos; those cannot be changed from "
             "this window. With a host set, Apply saves everything "
             "to the settings file but only reconnects; the engine "
             "fields take effect when a local engine is next "
             "started from this file.")

        head("This machine")
        hw = f"{os.cpu_count()} cores detected on this machine."
        rd = state.get("ready") or {}
        if rd:
            hw += (f" Engine reported: cores {rd.get('cores', '?')}, "
                   f"threads {rd.get('threads', '?')}, "
                   f"mode {rd.get('mode', '?')}, "
                   f"simd {'on' if rd.get('simd') == '1' else 'off'}, "
                   f"model {rd.get('model', '?')}.")
        else:
            hw += " The engine has not reported its READY line yet."
        note(hw)
        note(f"Settings file: {target}"
             + ("" if found_path else " (will be created by Apply)"))
        if ctx.get("cli_engine"):
            note("This session was launched with a command-line "
                 "--engine override. The fields above show the "
                 "settings file. Apply saves to the file and "
                 "restarts from it, replacing the override for "
                 "this session.")
        if ctx.get("cli_host") or ctx.get("cli_port"):
            note("This session was launched with command-line "
                 "--host / --port overrides; Apply restarts from "
                 "the settings file's remote values.")

        err = tk.Label(form, text="", bg=BG, fg="#f7768e", anchor="w",
                       justify="left", wraplength=430)
        err.grid(row=r, column=0, columnspan=2, sticky="w",
                 pady=(6, 2))
        r += 1
        btns = tk.Frame(form, bg=BG)
        btns.grid(row=r, column=0, columnspan=2, sticky="e",
                  pady=(4, 0))

        def gather():
            return {
                "engine": e_engine.get(),
                "checkpoint": e_ckpt.get(),
                "tokenizer": e_tok.get(),
                "mode": mode_var.get(),
                "threads": "" if auto_var.get() else e_threads.get(),
                "simd": bool(simd_var.get()),
                "max_new": e_max_new.get(),
                "max_pos": e_max_pos.get(),
                "host": e_host.get(),
                "port": e_port.get(),
            }

        def show_errors(errors):
            err.configure(text="\n".join(
                f"{k}: {v}" for k, v in errors.items()))

        def on_apply():
            nonlocal engine_cmd, host, port
            raw = gather()
            errors = validate_settings(raw)
            if errors:
                show_errors(errors)
                return
            if state["busy"] or state["turns"] > 0:
                if not messagebox.askyesno(
                        "Restart engine",
                        "Applying these settings restarts the "
                        "engine and clears the current conversation. "
                        "Continue?", parent=dlg):
                    return
            errors, result = apply_settings(raw, target)
            if errors:
                show_errors(errors)
                return
            append(chat, "[settings applied; engine restarting, "
                         "conversation cleared]\n", "sys")
            think.configure(state="normal")
            think.delete("1.0", "end")
            think.configure(state="disabled")
            dlg.destroy()
            # The applied settings become the session's current
            # settings, so a later Restart engine respawns with
            # them instead of reverting to the startup values
            # (Archie Round 4 P1-2, 2026-11-25).
            engine_cmd, host, port = (result["cmd"], result["host"],
                                      result["port"])
            connect(engine_cmd, host, port)

        tk.Button(btns, text="Apply", command=on_apply, bg=ACCENT,
                  fg="#16161e",
                  font=("TkDefaultFont", 10, "bold")).pack(
                      side="left", padx=(0, 6))
        tk.Button(btns, text="Cancel", command=dlg.destroy, bg=PANEL,
                  fg=FG).pack(side="left")
        try:
            dlg.grab_set()
        except Exception:
            pass

    connect(engine_cmd, host, port)

    draw_strip()
    poll()
    root.mainloop()


def load_ini(argv):
    """Read littlenemo.ini -> (engine_cmd, host, port), any of which
    may be None. engine_cmd is an argument LIST (see connect_process).
    Search order: --config PATH, ./littlenemo.ini,
    beside this script, the script's parent directory."""
    path = None
    if "--config" in argv:
        j = argv.index("--config")
        if j + 1 < len(argv):
            path = argv[j + 1]
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [path, os.path.join(os.getcwd(), "littlenemo.ini"),
                  os.path.join(here, "littlenemo.ini"),
                  os.path.join(here, "..", "littlenemo.ini")]
    found = None
    for cand in candidates:
        if cand and os.path.isfile(cand):
            found = cand
            break
    if not found:
        return None, None, None
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"))
    cfg.read(found, encoding="utf-8")
    base = os.path.dirname(os.path.abspath(found))

    def rel(p):
        p = (p or "").strip()
        if not p:
            return ""
        return p if os.path.isabs(p) else os.path.normpath(
            os.path.join(base, p))

    engine = rel(cfg.get("paths", "engine", fallback=""))
    ckpt = rel(cfg.get("paths", "checkpoint", fallback=""))
    tok = rel(cfg.get("paths", "tokenizer", fallback=""))
    cmd = None
    if engine and ckpt and tok:
        parts = [engine, ckpt, tok]
        max_new = cfg.get("engine", "max_new", fallback="").strip()
        if max_new:
            parts += ["--max-new", max_new]
        mode = cfg.get("engine", "mode", fallback="").strip()
        if mode:
            parts += ["--mode", mode]
        threads = cfg.get("engine", "threads", fallback="").strip()
        if threads and threads != "0":
            parts += ["--threads", threads]
        try:
            if cfg.getboolean("engine", "simd", fallback=False):
                parts += ["--simd"]
        except ValueError:
            pass
        max_pos = cfg.get("engine", "max_pos", fallback="").strip()
        if max_pos:
            parts += ["--max-pos", max_pos]
        cmd = parts
    host = cfg.get("remote", "host", fallback="").strip() or None
    port = cfg.get("remote", "port", fallback="").strip() or None
    return cmd, host, port


# ------------------------------------------------------------- settings ---
#
# The Settings menu's code path, kept free of Tk so it is
# headless-testable (tests/test_gui_settings.py). The panel works
# on raw field strings; parse_settings turns them into typed
# values plus a per-field error dict; apply_settings is the one
# path that writes: it validates first and writes nothing at all
# when any field is wrong.

MODES = ("minimum", "medium", "max", "ultra")

# Engine built-in defaults (littlenemo/src/engine.c), used only
# to label what a blank field means in the panel.
ENGINE_DEFAULTS = {"mode": "max", "max_new": 256, "max_pos": 4096}


def find_ini_path(argv):
    """The littlenemo.ini the launcher would read, or None.
    Same search order as load_ini: --config PATH,
    ./littlenemo.ini, beside this script, the script's parent."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [_config_arg(argv),
                  os.path.join(os.getcwd(), "littlenemo.ini"),
                  os.path.join(here, "littlenemo.ini"),
                  os.path.join(here, "..", "littlenemo.ini")]
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def _config_arg(argv):
    if "--config" in argv:
        j = argv.index("--config")
        if j + 1 < len(argv):
            return argv[j + 1]
    return None


def resolve_settings(argv):
    """Read the INI the launcher would use -> (values, ini_path).
    values holds the raw INI-level settings as strings (simd as a
    bool): engine, checkpoint, tokenizer, mode, threads, max_new,
    max_pos, host, port. Unset keys come back as "" (simd False).
    ini_path is None when no INI file exists. These are the
    effective INI values; command-line overrides are tracked
    separately by main() and only reported by the panel."""
    values = {"engine": "", "checkpoint": "", "tokenizer": "",
              "mode": "", "threads": "", "simd": False,
              "max_new": "", "max_pos": "", "host": "", "port": ""}
    path = find_ini_path(argv)
    if not path:
        return values, None
    cfg = configparser.ConfigParser(
        inline_comment_prefixes=(";", "#"))
    cfg.read(path, encoding="utf-8")
    for key in ("engine", "checkpoint", "tokenizer"):
        values[key] = cfg.get("paths", key, fallback="").strip()
    for key in ("mode", "threads", "max_new", "max_pos"):
        values[key] = cfg.get("engine", key, fallback="").strip()
    try:
        values["simd"] = cfg.getboolean("engine", "simd",
                                         fallback=False)
    except ValueError:
        values["simd"] = False
    values["host"] = cfg.get("remote", "host", fallback="").strip()
    values["port"] = cfg.get("remote", "port", fallback="").strip()
    return values, path


def _parse_int_field(raw, key, minimum, errors):
    """One integer panel field: blank -> None (unset / auto);
    otherwise an int, with an error recorded when it is not a
    whole number or is below the minimum."""
    text = (raw.get(key) or "").strip()
    if not text:
        return None
    try:
        value = int(text, 10)
    except ValueError:
        errors[key] = f"{key} must be a whole number"
        return None
    if value < minimum:
        errors[key] = f"{key} must be at least {minimum}"
        return None
    return value


def parse_settings(raw):
    """Panel raw strings -> (parsed, errors). parsed is the typed
    settings dict (threads None means auto: the mode decides;
    max_new / max_pos / port None means unset); errors maps field
    name -> message and is empty when the settings are valid."""
    errors = {}
    parsed = {}
    for key in ("engine", "checkpoint", "tokenizer", "host"):
        parsed[key] = (raw.get(key) or "").strip()
    mode = (raw.get("mode") or "").strip()
    if mode not in MODES:
        errors["mode"] = "mode must be one of: " + ", ".join(MODES)
    parsed["mode"] = mode
    if (raw.get("threads") or "").strip() in ("", "0"):
        parsed["threads"] = None  # 0 / blank: the mode decides
    else:
        parsed["threads"] = _parse_int_field(raw, "threads", 1,
                                              errors)
    parsed["simd"] = bool(raw.get("simd"))
    parsed["max_new"] = _parse_int_field(raw, "max_new", 1, errors)
    parsed["max_pos"] = _parse_int_field(raw, "max_pos", 1, errors)
    port = _parse_int_field(raw, "port", 1, errors)
    if port is not None and port > 65535:
        errors["port"] = "port must be between 1 and 65535"
        port = None
    parsed["port"] = port
    if parsed["host"] and parsed["port"] is None \
            and "port" not in errors:
        errors["port"] = "port is required when a host is set"
    if parsed["port"] is not None and not parsed["host"]:
        errors["host"] = "host is required when a port is set"
    if not parsed["host"]:
        for key in ("engine", "checkpoint", "tokenizer"):
            if not parsed[key]:
                errors[key] = (f"{key} path is required for a "
                               f"local engine")
    return parsed, errors


def validate_settings(raw):
    """Errors dict for panel raw strings; empty means valid."""
    return parse_settings(raw)[1]


def update_ini_text(text, updates):
    """Apply (section, key, value) updates to INI text line by
    line, so comments, unrelated sections and unrelated keys are
    preserved exactly. A key that exists in its section is
    rewritten in place, keeping the line's original key text and
    spacing up to the value; a missing key is added at the end of
    its section; a missing section is appended at the end."""
    pending = {}
    for section, key, value in updates:
        pending[(section.lower(), key.lower())] = (section, key,
                                                   value)
    header_re = re.compile(r"^\s*\[(.+)\]\s*$")
    key_re = re.compile(r"^(\s*)([A-Za-z0-9_.\-]+)(\s*[=:]\s*)(.*)$")
    out = []
    done = set()
    seen_sections = set()
    section = None

    def flush_missing(sec):
        if sec is None:
            return
        for (s, k), (s0, k0, v) in pending.items():
            if s == sec.lower() and (s, k) not in done:
                out.append(f"{k0} = {v}")
                done.add((s, k))

    lines = text.split("\n") if text else []
    if lines and lines[-1] == "":
        lines.pop()  # a trailing newline is a terminator, not a line
    for line in lines:
        m = header_re.match(line)
        if m:
            flush_missing(section)
            section = m.group(1).strip()
            seen_sections.add(section.lower())
            out.append(line)
            continue
        km = key_re.match(line)
        if km and section is not None:
            ident = (section.lower(), km.group(2).lower())
            if ident in pending and ident not in done:
                out.append(km.group(1) + km.group(2) + km.group(3)
                           + pending[ident][2])
                done.add(ident)
                continue
        out.append(line)
    flush_missing(section)
    for (s, k), (s0, k0, v) in pending.items():
        if s not in seen_sections and (s, k) not in done:
            out.append(f"[{s0}]")
            seen_sections.add(s)
            flush_missing(s0)
    return "\n".join(out) + "\n"


def _resolve_against(base, p):
    p = (p or "").strip()
    if not p:
        return ""
    return p if os.path.isabs(p) else os.path.normpath(
        os.path.join(base, p))


def _cmd_from_parsed(parsed, base):
    """The engine argument list for parsed settings, built in the
    same order load_ini builds it from the INI."""
    engine = _resolve_against(base, parsed["engine"])
    ckpt = _resolve_against(base, parsed["checkpoint"])
    tok = _resolve_against(base, parsed["tokenizer"])
    if not (engine and ckpt and tok):
        return None
    parts = [engine, ckpt, tok]
    if parsed["max_new"] is not None:
        parts += ["--max-new", str(parsed["max_new"])]
    if parsed["mode"]:
        parts += ["--mode", parsed["mode"]]
    if parsed["threads"] is not None:
        parts += ["--threads", str(parsed["threads"])]
    if parsed["simd"]:
        parts += ["--simd"]
    if parsed["max_pos"] is not None:
        parts += ["--max-pos", str(parsed["max_pos"])]
    return parts


def apply_settings(raw, ini_path):
    """The settings panel's Apply, as one code path: validate the
    raw field strings; on any error return (errors, None) with
    nothing written. On success write the values into the INI at
    ini_path (creating it if needed, preserving unrelated
    content) and return ({}, result), where result carries the
    engine argument list and the remote target to respawn with,
    plus the parsed values."""
    parsed, errors = parse_settings(raw)
    if errors:
        return errors, None
    updates = [
        ("paths", "engine", parsed["engine"]),
        ("paths", "checkpoint", parsed["checkpoint"]),
        ("paths", "tokenizer", parsed["tokenizer"]),
        ("engine", "mode", parsed["mode"]),
        ("engine", "threads",
         "0" if parsed["threads"] is None else str(parsed["threads"])),
        ("engine", "simd", "true" if parsed["simd"] else "false"),
        ("engine", "max_new",
         "" if parsed["max_new"] is None else str(parsed["max_new"])),
        ("engine", "max_pos",
         "" if parsed["max_pos"] is None else str(parsed["max_pos"])),
        ("remote", "host", parsed["host"]),
        ("remote", "port",
         "" if parsed["port"] is None else str(parsed["port"])),
    ]
    text = ""
    if ini_path and os.path.isfile(ini_path):
        with open(ini_path, "r", encoding="utf-8") as f:
            text = f.read()
    tmp = ini_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(update_ini_text(text, updates))
    os.replace(tmp, ini_path)
    base = os.path.dirname(os.path.abspath(ini_path))
    result = {
        "cmd": _cmd_from_parsed(parsed, base),
        "host": parsed["host"] or None,
        "port": (str(parsed["port"])
                 if parsed["port"] is not None else None),
        "parsed": parsed,
    }
    return {}, result


def main(argv):
    engine_cmd, host, port = load_ini(argv)
    cli_engine = cli_host = cli_port = None
    i = 0
    while i < len(argv):
        if argv[i] == "--engine":
            engine_cmd = argv[i + 1]
            cli_engine = argv[i + 1]
            i += 2
        elif argv[i] == "--host":
            host = argv[i + 1]
            cli_host = argv[i + 1]
            i += 2
        elif argv[i] == "--port":
            port = argv[i + 1]
            cli_port = argv[i + 1]
            i += 2
        elif argv[i] == "--config":
            i += 2
        else:
            i += 1
    settings_ctx = {
        "argv": list(argv),
        "ini_path": find_ini_path(argv),
        "config_arg": _config_arg(argv),
        "cli_engine": cli_engine,
        "cli_host": cli_host,
        "cli_port": cli_port,
    }
    run_gui(engine_cmd, host, port, settings_ctx)


if __name__ == "__main__":
    main(sys.argv[1:])
