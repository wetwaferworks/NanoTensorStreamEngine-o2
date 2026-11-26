#!/usr/bin/env python3
"""nemo_chat.py - NTSE chat front end (SuperNemo and Little Nemo).

The ONLY Python piece of NTSE's chat path, by JD's standing rule: the
front end is a Python GUI that calls the pure-C backend. This module is
the entry point and the Tk-free half: the protocol (PROTOCOL.md v2 plus
the backward-compatible v3 extensions), the engine client, the model
list and send gating, and the settings code path. The window lives in
nemo_ui.py, attachments in nemo_media.py, the stats panel's data and
the thinking-line headline in nemo_stats.py, scripted walk-throughs in
nemo_demo.py. It speaks to the engine over a subprocess pipe or a TCP
socket (so the same window works against an engine on your own machine
or one running on Elfie's computer).

    python3 nemo_chat.py [--engine "../nemo --serve-stdio"] [--host H --port P]
                         [--model supernemo|littlenemo]
                         [--theme pixel|lattice|stream|handshake|owl]
                         [--supernemo-engine CMD | --supernemo-host H
                          --supernemo-port P]

Models: SuperNemo is the code name for Nemotron 3.5 Super VL
120B-A12B (text, image and video by design). Its NTSE engine path
(512-expert MoE plus the RADIO vision tower) is not written yet, so
with no [supernemo] engine configured the window says "SuperNemo
engine not available yet" and sends nothing. Little Nemo (Nemotron
Nano 9B v2) is text-only; attachments are refused before anything
is sent. Test fixture: ../tests/fake_engine.py (replies say so).

Configuration can also come from an INI file, so the window can be
launched by clicking with no arguments at all. The GUI looks for
littlenemo.ini in the current directory, beside this script, and in
the script's parent directory (or at the path given by --config):

    [paths]
    engine     = ../bin/nemo-engine.exe
    checkpoint = <models-dir>/Nemotron-Nano-9B-v2
    tokenizer  = <models-dir>/Nemotron-Nano-9B-v2/tokenizer.json

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


# ------------------------------------------------- protocol v3 additions ---
#
# Everything below is additive and backward compatible (PROTOCOL.md,
# "v3 extensions"): a v2 engine never sends the new lines and the GUI
# works without them; a v2 client treats the new lines as 'raw' and
# ignores them. Values of the new key=value lines are percent-encoded
# (RFC 3986, every byte outside A-Z a-z 0-9 - . _ ~), so names and paths
# may carry spaces, '=' and non-ASCII text on a whitespace-split line.

# A protocol line longer than this is not parsed (a hostile or broken
# peer cannot make the reader build megabyte payloads); it surfaces as
# ('raw', <truncated>) and is ignored by the GUI.
MAX_LINE = 256 * 1024

# Attachment bounds the client enforces before it sends anything; the
# engine validates again on its side (see PROTOCOL.md).
MAX_ATTACHMENTS = 8
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024
MAX_IMAGE_SIDE = 16384
MAX_IMAGE_PIXELS = 100_000_000
MAX_FRAMES = 32
MAX_PATH_CHARS = 4096
MEDIA_TYPES = ("image", "video")

# Keys whose values stay strings even when they look numeric.
_STRING_KEYS = {
    "cpu", "gpu", "driver", "igpu", "accel", "os", "name", "tensor",
    "lane", "state", "reason", "mode", "model", "path", "mime", "type",
    "experts", "kind", "weights", "media", "cuda", "opencl", "note",
    "device", "codec", "phase",
}


def pct_encode(value):
    from urllib.parse import quote

    return quote(str(value), safe="")


def pct_decode(value):
    from urllib.parse import unquote

    return unquote(value)


def _num(v):
    """'12' -> 12, '1.5' -> 1.5, anything else unchanged."""
    try:
        return int(v, 10)
    except (TypeError, ValueError):
        pass
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    if f != f or f in (float("inf"), float("-inf")):
        return v  # nan / inf stay text; never feed them to a readout
    return f


def _typed_kv(tokens):
    """key=value tokens -> dict with percent-decoded values; numeric
    values become int/float except for keys in _STRING_KEYS."""
    d = {}
    for k, v in _kv(tokens).items():
        v = pct_decode(v)
        d[k] = v if k in _STRING_KEYS else _num(v)
    return d


# v3 telemetry verbs -> event kinds (engine to client).
_EXT_VERBS = {
    "HW": "hw",            # hardware the engine runs on
    "PREFILL": "prefill",  # prompt processing summary
    "EXPAND": "expand",    # a tensor being expanded (dequantized)
    "DTC": "dtc",          # a tensor in dense tensor compute
    "HS": "hs",            # Tensor Handshake / prefetch counters
    "MEM": "mem",          # RAM / VRAM in use
    "MOE": "moe",          # mixture-of-experts routing for a layer
    "STAT": "stat",        # any other engine statistic
    "MEDIA_OK": "media_ok",
    "MEDIA_ERR": "media_err",
}


def _parse_line_strict(line):
    if line.startswith("READY "):
        return ("ready", {k: pct_decode(v)
                          for k, v in _kv(line.split()[2:]).items()})
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
        # v3: any further STATS keys (prompt_tokens, ttft_ms, ...)
        for k, v in list(d.items()):
            if isinstance(v, str):
                d[k] = _num(pct_decode(v))
        return ("stats", d)
    if line == "DONE":
        return ("done", None)
    if line.startswith("ERROR "):
        return ("error", line[6:])
    verb, _, rest = line.partition(" ")
    if verb in _EXT_VERBS:
        return (_EXT_VERBS[verb], _typed_kv(rest.split()))
    return ("raw", line)


def parse_line(line):
    """One protocol line -> (kind, payload). v2 kinds: ready, think,
    tok, layer, tick, stats, done, error. v3 kinds: hw, prefill,
    expand, dtc, hs, mem, moe, stat, media_ok, media_err. Unknown or
    malformed lines are ('raw', line): a bad line from the engine is
    ignored instead of killing the reader thread."""
    line = line.rstrip("\r\n")
    if len(line) > MAX_LINE:
        return ("raw", line[:256])
    try:
        return _parse_line_strict(line)
    except (ValueError, IndexError):
        return ("raw", line)


def ready_media(ready):
    """The media types an engine accepts, from its READY line's v3
    'media=' key. A v2 READY has no such key: text only."""
    raw = (ready or {}).get("media", "")
    return {m for m in str(raw).split(",") if m in MEDIA_TYPES}


def format_media_lines(att_id, att):
    """The client->engine lines announcing one attachment (PROTOCOL.md
    v3): a MEDIA line, then one FRAME line per extracted video frame.
    att is the dict nemo_media.inspect_attachment returns (plus an
    optional 'frames_list' of {'path', 'ts_ms'})."""
    if att.get("type") not in MEDIA_TYPES:
        raise ValueError("attachment type must be image or video")
    path = str(att["path"])
    if not path or len(path) > MAX_PATH_CHARS:
        raise ValueError("attachment path is empty or too long")
    fields = [("id", int(att_id)), ("type", att["type"]),
              ("mime", att.get("mime") or "application/octet-stream"),
              ("bytes", int(att["bytes"]))]
    for k in ("w", "h", "frames", "dur_ms"):
        if att.get(k) not in (None, ""):
            fields.append((k, int(att[k])))
    frames = list(att.get("frames_list") or [])[:MAX_FRAMES]
    if frames:
        fields.append(("sent_frames", len(frames)))
    fields.append(("name", att.get("name") or os.path.basename(path)))
    fields.append(("path", path))
    lines = ["MEDIA " + " ".join(f"{k}={pct_encode(v)}" for k, v in fields)]
    for i, fr in enumerate(frames):
        lines.append(f"FRAME id={int(att_id)} idx={i} "
                     f"ts_ms={int(fr.get('ts_ms', 0))} "
                     f"path={pct_encode(fr['path'])}")
    return lines


class NemoClient:
    """Speaks the protocol to one engine. on_event(kind, payload) is
    called from the reader thread; the GUI marshals to its own thread
    through a queue. Sends are made from one thread only (the GUI's
    Tk thread, or a test's main thread), so no lock is needed: the
    reader thread only reads, and nothing else writes."""

    def __init__(self, on_event):
        self.on_event = on_event
        self._w = None
        self._proc = None
        self._sock = None
        self._next_media_id = 1

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
        if self._w is None:
            raise RuntimeError("not connected")
        self._w.write(line + "\n")
        self._w.flush()

    def send_prompt(self, text):
        self._send("PROMPT " + escape(text))

    def send_turn(self, text, attachments=()):
        """One user turn with optional attachments (v3): the MEDIA /
        FRAME lines for each attachment, then the PROMPT they belong
        to. Returns the media ids used, in order. Only call this with
        attachments for an engine whose READY advertised them
        (ready_media); check_send enforces that in the GUI."""
        attachments = list(attachments)
        if len(attachments) > MAX_ATTACHMENTS:
            raise ValueError(f"at most {MAX_ATTACHMENTS} attachments per turn")
        lines = []
        ids = []
        for att in attachments:
            att_id = self._next_media_id
            self._next_media_id += 1
            lines.extend(format_media_lines(att_id, att))
            ids.append(att_id)
        for line in lines:
            self._send(line)
        self.send_prompt(text)
        return ids

    def stop(self):
        self._send("STOP")

    def reset(self):
        self._send("RESET")

    def quit(self):
        try:
            self._send("QUIT")
        except Exception:
            pass

    def close(self):
        """Retire this client: QUIT, then end the process / socket."""
        self.quit()
        try:
            if self._proc is not None:
                self._proc.terminate()
        except Exception:
            pass
        try:
            if self._sock is not None:
                self._sock.close()
        except Exception:
            pass
        self._w = None


# --------------------------------------------------------------- models ---
#
# The models this window can talk to. SuperNemo is the code name for
# Nemotron 3.5 Super VL 120B-A12B; its NTSE engine path (512-expert MoE
# plus the RADIO vision tower) is not written yet, so unless an engine
# is configured for it the GUI says so plainly and sends nothing.

MODELS = {
    "supernemo": {
        "label": "SuperNemo",
        "full": "Nemotron 3.5 Super VL 120B-A12B",
        "inputs": ("text", "image", "video"),
        "unavailable": ("SuperNemo engine not available yet. NTSE has no "
                        "engine path for its 512-expert MoE and RADIO "
                        "vision tower so far; nothing is sent and no reply "
                        "is invented. Little Nemo is available for text."),
    },
    "littlenemo": {
        "label": "Little Nemo",
        "full": "Nemotron Nano 9B v2",
        "inputs": ("text",),
        "unavailable": ("No Little Nemo engine is configured (see "
                        "littlenemo.ini [paths] or --engine / --host)."),
    },
}
MODEL_ORDER = ("supernemo", "littlenemo")


def resolve_model_targets(argv):
    """Engine targets per model -> {key: {'cmd', 'host', 'port'}}.
    Little Nemo comes from littlenemo.ini [paths]/[remote] and
    --engine/--host/--port, exactly as before. SuperNemo comes from an
    optional [supernemo] section (engine = full command line, or
    host/port) and --supernemo-engine / --supernemo-host /
    --supernemo-port. A model with no target is shown as unavailable."""
    cmd, host, port = load_ini(argv)
    s_cmd = s_host = s_port = None
    path = find_ini_path(argv)
    if path:
        cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
        cfg.read(path, encoding="utf-8")
        s_cmd = cfg.get("supernemo", "engine", fallback="").strip() or None
        s_host = cfg.get("supernemo", "host", fallback="").strip() or None
        s_port = cfg.get("supernemo", "port", fallback="").strip() or None
    i = 0
    while i < len(argv):
        a = argv[i]
        val = argv[i + 1] if i + 1 < len(argv) else None
        if a == "--engine":
            cmd = val
        elif a == "--host":
            host = val
        elif a == "--port":
            port = val
        elif a == "--supernemo-engine":
            s_cmd = val
        elif a == "--supernemo-host":
            s_host = val
        elif a == "--supernemo-port":
            s_port = val
        else:
            i += 1
            continue
        i += 2
    return {
        "littlenemo": {"cmd": cmd, "host": host, "port": port},
        "supernemo": {"cmd": s_cmd, "host": s_host if s_port else None,
                      "port": s_port if s_host else None},
    }


def target_available(target):
    t = target or {}
    return bool(t.get("cmd") or (t.get("host") and t.get("port")))


def check_send(model_key, available, connected, ready, text, attachments,
               models=None):
    """Whether a turn may go out -> (ok, message). Pure, so the
    refusals are unit-tested: an unavailable SuperNemo engine, a
    disconnected engine, attachments the engine did not advertise
    (Little Nemo is text-only), too many attachments, an empty turn."""
    info = (models or MODELS).get(
        model_key, {"label": model_key, "unavailable": ""})
    attachments = list(attachments or [])
    if not available:
        return False, info["unavailable"]
    if not connected:
        return False, f"{info['label']} engine is not connected yet."
    if not (text or "").strip() and not attachments:
        return False, ""
    if len(attachments) > MAX_ATTACHMENTS:
        return False, (f"At most {MAX_ATTACHMENTS} attachments per turn; "
                       f"remove {len(attachments) - MAX_ATTACHMENTS}.")
    accepted = ready_media(ready)
    refused = sorted({a.get("type") for a in attachments} - accepted)
    if refused:
        if not accepted:
            return False, (f"{info['label']} is text-only: the attachments "
                           f"were not sent. Remove them, or switch to "
                           f"SuperNemo for image and video.")
        return False, (f"This engine does not accept {', '.join(refused)} "
                       f"attachments (it accepts "
                       f"{', '.join(sorted(accepted))}); nothing was sent.")
    if not (text or "").strip():
        return False, "Add a message to go with the attachments."
    return True, ""


# ---------------------------------------------------------------- GUI ---
#
# The window itself lives in nemo_ui.py (Tk); everything above and
# below this point is Tk-free and headless-testable.


def run_gui(engine_cmd=None, host=None, port=None, settings_ctx=None):
    """Compatibility entry point (v2 signature): open the chat window
    with Little Nemo on the given engine target, SuperNemo per the
    settings file."""
    argv = (settings_ctx or {}).get("argv", [])
    targets = resolve_model_targets(argv)
    targets["littlenemo"] = {"cmd": engine_cmd, "host": host, "port": port}
    import nemo_ui

    nemo_ui.run_app(targets, "littlenemo", settings_ctx or {})


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



def _arg(argv, flag):
    if flag in argv:
        j = argv.index(flag)
        if j + 1 < len(argv):
            return argv[j + 1]
    return None


def main(argv):
    targets = resolve_model_targets(argv)
    model = _arg(argv, "--model") or "littlenemo"
    if model not in MODELS:
        sys.stderr.write(f"unknown --model {model!r}; choose one of: "
                         f"{', '.join(MODEL_ORDER)}\n")
        return 2
    lt = targets["littlenemo"]
    settings_ctx = {
        "argv": list(argv),
        "ini_path": find_ini_path(argv),
        "config_arg": _config_arg(argv),
        "cli_engine": _arg(argv, "--engine"),
        "cli_host": _arg(argv, "--host"),
        "cli_port": _arg(argv, "--port"),
        "cli_theme": _arg(argv, "--theme"),
        "engine_cmd": lt["cmd"], "host": lt["host"], "port": lt["port"],
    }
    import nemo_ui

    return nemo_ui.run_app(targets, model, settings_ctx,
                           demo=_arg(argv, "--demo"),
                           shots=_arg(argv, "--shots"))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
