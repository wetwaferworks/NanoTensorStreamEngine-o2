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

The client half (NemoClient) has no Tk dependency and is unit-testable
headlessly; see tests/test_gui_client.py.
"""

import configparser
import os
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
        self._proc = subprocess.Popen(
            cmd, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, encoding="utf-8", bufsize=1,
        )
        self._w = self._proc.stdin
        self._start_reader(self._proc.stdout)

    def connect_tcp(self, host, port):
        import socket

        self._sock = socket.create_connection((host, port), timeout=10)
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


def run_gui(engine_cmd=None, host=None, port=None):
    import queue
    import time
    import tkinter as tk
    from tkinter.scrolledtext import ScrolledText

    events = queue.Queue()
    client = NemoClient(lambda k, p: events.put((k, p)))

    root = tk.Tk()
    root.title("Little Nemo")
    root.configure(bg=BG)
    root.geometry("1060x760")

    state = {
        "mode": "idle", "layer": None, "next": None, "active": 0,
        "staged": 0, "tok": 0, "ms_tok": 0.0, "gbs": 0.0,
        "turn_start": None, "busy": False, "model": "...",
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
               if lyr is not None else "layer -")
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
            client.send_prompt(text)
        except Exception as exc:
            append(chat, f"[not connected: {exc}]\n", "sys")
            state["busy"] = False
            set_mode("idle")

    def on_stop():
        try:
            client.stop()
        except Exception:
            pass

    def on_reset():
        try:
            client.reset()
        except Exception:
            pass
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
                kind, payload = events.get_nowait()
                if kind == "ready":
                    state["model"] = payload.get("model", "?")
                    status.configure(
                        text=f"connected · {state['model']} · "
                             f"weights {payload.get('weights', '?')}")
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

    if host:
        try:
            client.connect_tcp(host, int(port))
        except Exception as exc:
            status.configure(text=f"connect failed: {exc}")
    elif engine_cmd:
        try:
            client.connect_process(engine_cmd)
        except Exception as exc:
            status.configure(text=f"engine failed to start: {exc}")
    else:
        status.configure(text="no engine configured (see littlenemo.ini or --engine / --host)")

    draw_strip()
    poll()
    root.mainloop()


def _quote(part):
    if part and all(c not in part for c in ' \t"'):
        return part
    return '"' + part.replace('"', '\\"') + '"'


def load_ini(argv):
    """Read littlenemo.ini -> (engine_cmd, host, port), any of which
    may be None. Search order: --config PATH, ./littlenemo.ini,
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
    cfg = configparser.ConfigParser()
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
        parts = [_quote(engine), _quote(ckpt), _quote(tok)]
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
        cmd = " ".join(parts)
    host = cfg.get("remote", "host", fallback="").strip() or None
    port = cfg.get("remote", "port", fallback="").strip() or None
    return cmd, host, port


def main(argv):
    engine_cmd, host, port = load_ini(argv)
    i = 0
    while i < len(argv):
        if argv[i] == "--engine":
            engine_cmd = argv[i + 1]
            i += 2
        elif argv[i] == "--host":
            host = argv[i + 1]
            i += 2
        elif argv[i] == "--port":
            port = argv[i + 1]
            i += 2
        elif argv[i] == "--config":
            i += 2
        else:
            i += 1
    run_gui(engine_cmd, host, port)


if __name__ == "__main__":
    main(sys.argv[1:])
