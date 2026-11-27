# nq2_quant.py ... the logic behind the Quantize dialog of nq2-gui.py.
#
# Python front end only, under JD's ruling that Python may render and relay:
# every byte of quantizing is done by the C converter (nq2-quantize,
# Elfie's). This module builds its command line, checks the options the way
# the converter does (so a mistake is caught before a long run), launches it
# without a console window, follows its progress lines and reports the
# result. It does no model work and never downloads anything.
#
# Direction by JD. Converter by Elfie. Co-implementation by Cortana, built
# with Grok Bot by SpaceX.

import hashlib
import json
import os
import re
import subprocess
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
EXE_NAME = "nq2-quantize.exe" if os.name == "nt" else "nq2-quantize"
CREATE_NO_WINDOW = 0x08000000
CREATE_NEW_CONSOLE = 0x00000010
MAX_THREADS = 12

# Converter defaults (tools/nq2-quantize.c, main). An option left at its
# default is not passed, so the command line shows only what was chosen.
DEFAULTS = {
    "block": 256,
    "entries": 256,
    "budget": 250000000,
    "filter": "",
    "sample_blocks": 32768,
    "tensor_sample_blocks": 8192,
    "iters": 8,
    "books": "global",        # global | per-tensor
    "norm": "minmax",         # minmax | rms
    "format": 3,              # 1 | 2 | 3
    "pq": 0,                  # 0 (off) | 4 | 8
    "budgets": False,
    "embed_fp16": False,
    "residual": False,
}

# Named recipes. The CUDA pack is the one documented in docs/ACCEL-CUDA.md.
PRESETS = {
    "Default (one global book)": {},
    "Per-tensor books (rms)": {"books": "per-tensor", "norm": "rms"},
    "CUDA pack (per-tensor, rms, fp16 embeddings, PQ4)": {
        "books": "per-tensor", "norm": "rms", "embed_fp16": True, "pq": 4},
}


def default_threads():
    """All logical CPUs, up to the 12 the dialog offers."""
    return max(1, min(MAX_THREADS, os.cpu_count() or 1))


def new_options():
    o = dict(DEFAULTS)
    o["threads"] = default_threads()
    return o


def preset_options(name, threads=None):
    o = new_options()
    o.update(PRESETS[name])
    if threads is not None:
        o["threads"] = threads
    return o


def find_converter(explicit=""):
    """Find nq2-quantize: the configured path, NQ2_QUANTIZE, beside this
    script, the repository root, then PATH. Returns a path or None."""
    cands = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get("NQ2_QUANTIZE", "")
    if env:
        cands.append(env)
    cands.append(os.path.join(HERE, EXE_NAME))
    cands.append(os.path.join(REPO, EXE_NAME))
    for c in cands:
        if os.path.isfile(c):
            return c
    for d in os.environ.get("PATH", "").split(os.pathsep):
        c = os.path.join(d.strip('"'), EXE_NAME)
        if d and os.path.isfile(c):
            return c
    return None


def collect_inputs(path):
    """A .safetensors file, or every .safetensors in a folder (sorted by
    name, not recursive). Returns a list of paths."""
    if os.path.isdir(path):
        names = sorted(n for n in os.listdir(path)
                       if n.lower().endswith(".safetensors"))
        return [os.path.join(path, n) for n in names]
    return [path] if os.path.isfile(path) else []


def _inside(child, parent):
    c = os.path.normcase(os.path.abspath(child))
    p = os.path.normcase(os.path.abspath(parent))
    return c == p or c.startswith(p + os.sep)


def validate(inputs, out_prefix, opts):
    """Return (errors, warnings). The option rules mirror nq2-quantize."""
    errs, warns = [], []
    if not inputs:
        errs.append("Pick a .safetensors file, or a folder that holds one "
                    "or more.")
    if not out_prefix:
        errs.append("Choose an output path (folder and name prefix).")
    else:
        d = os.path.dirname(os.path.abspath(out_prefix))
        if not os.path.isdir(d):
            errs.append("The output folder does not exist: %s" % d)
        elif _inside(d, REPO):
            warns.append("The output folder is inside the repository. "
                         "Packs are large and may come from private weights; "
                         "keep them out of the repo.")
    try:
        t = int(opts.get("threads", 0))
        if not 1 <= t <= MAX_THREADS:
            errs.append("Threads must be 1 to %d." % MAX_THREADS)
    except (TypeError, ValueError):
        errs.append("Threads must be a whole number, 1 to %d." % MAX_THREADS)

    def num(key, label, lo=1, hi=None):
        try:
            v = int(opts.get(key))
        except (TypeError, ValueError):
            errs.append("%s must be a whole number." % label)
            return None
        if v < lo or (hi is not None and v > hi):
            errs.append("%s must be %d to %d." % (label, lo, hi)
                        if hi is not None else
                        "%s must be at least %d." % (label, lo))
            return None
        return v

    block = num("block", "Block size", 1, 4096)
    num("entries", "Dictionary entries", 1, 256)
    num("budget", "Weight budget", 1)
    num("sample_blocks", "Sample blocks", 1)
    num("tensor_sample_blocks", "Tensor sample blocks", 1)
    num("iters", "k-means iterations", 1)

    books = opts.get("books")
    fmt = opts.get("format")
    pq = int(opts.get("pq") or 0)
    v3only = bool(pq or opts.get("budgets") or opts.get("embed_fp16"))
    if books not in ("global", "per-tensor"):
        errs.append("Books must be global or per-tensor.")
    if fmt not in (1, 2, 3):
        errs.append("Format must be 1, 2 or 3.")
    if v3only and books != "per-tensor":
        errs.append("PQ, per-tensor budgets and fp16 embeddings need "
                    "per-tensor books.")
    if v3only and fmt != 3:
        errs.append("PQ, per-tensor budgets and fp16 embeddings are format 3 "
                    "options.")
    if books == "per-tensor" and fmt == 1:
        errs.append("Per-tensor books are format 2 or 3.")
    if books == "global" and fmt == 2:
        errs.append("The global-book layout is format 1 or 3.")
    if pq not in (0, 4, 8):
        errs.append("PQ subvector dimension must be 4 or 8.")
    if pq and block is not None and block != 256:
        errs.append("PQ is defined at block 256.")
    if pq and opts.get("budgets"):
        errs.append("PQ and per-tensor budgets cannot be combined.")
    return errs, warns


def build_command(exe, inputs, out_prefix, opts):
    """The converter's argv: only options that differ from its defaults,
    and always an explicit --threads N."""
    cmd = [exe, "--out", out_prefix] + list(inputs)
    flags = (("block", "--block"), ("entries", "--entries"),
             ("budget", "--budget"), ("sample_blocks", "--sample-blocks"),
             ("tensor_sample_blocks", "--tensor-sample-blocks"),
             ("iters", "--iters"), ("books", "--books"), ("norm", "--norm"),
             ("format", "--format"))
    for key, flag in flags:
        v = opts.get(key, DEFAULTS[key])
        if v != DEFAULTS[key]:
            cmd += [flag, str(v)]
    if str(opts.get("filter", "")).strip():
        cmd += ["--filter", str(opts["filter"]).strip()]
    if opts.get("pq"):
        cmd += ["--pq", str(int(opts["pq"]))]
    for key, flag in (("budgets", "--budgets"), ("embed_fp16", "--embed-fp16"),
                      ("residual", "--residual")):
        if opts.get(key):
            cmd.append(flag)
    cmd += ["--threads", str(int(opts["threads"]))]
    return cmd


def quote_command(cmd):
    """The command as one line a person can paste into a shell."""
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    import shlex
    return " ".join(shlex.quote(c) for c in cmd)


_PCT = re.compile(r"\((\d+(?:\.\d+)?)%\)")


def parse_progress(line):
    """Percent complete (0..100) from a converter progress line, or None.
    The converter flushes '  tensors D / N (P%)' and '  pass 2: c / n
    chunks (P%)' lines as it works."""
    if "(" not in line:
        return None
    m = _PCT.search(line)
    if m and ("tensors" in line or "chunks" in line):
        return max(0.0, min(100.0, float(m.group(1))))
    return None


def phase_of(line):
    """A short phase label from a converter line, or None."""
    s = line.strip()
    if s.startswith("pass 1"):
        return "sampled"
    if s.startswith("k-means"):
        return "dictionary learned"
    if s.startswith("parallel per-tensor"):
        return "quantizing per-tensor"
    if s.startswith("pass 2"):
        return "packing"
    return None


def output_files(out_prefix):
    """The files the converter publishes for a prefix (those that exist)."""
    names = [out_prefix + ext for ext in
             (".nq2", ".nq2book", ".nq2manifest", ".report.txt")]
    return [p for p in names if os.path.isfile(p)]


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def hash_outputs(out_prefix):
    """SHA-256 of the pack files, keyed by extension (.nq2, .nq2book,
    .nq2manifest), so runs with different output names compare directly.
    The report is left out: it holds timings."""
    return {p[len(out_prefix):]: sha256_file(p) for p in output_files(out_prefix)
            if not p.endswith(".report.txt")}


def process_flags(show_console):
    """subprocess creation flags: hidden by default, own console if asked."""
    if os.name != "nt":
        return 0
    return CREATE_NEW_CONSOLE if show_console else CREATE_NO_WINDOW


class Job:
    """One converter run. Output lines go to on_line(text); progress to
    on_progress(percent_or_None, phase); on_done(returncode, message) once.
    Callbacks arrive on a reader thread: marshal to the UI thread."""

    def __init__(self, cmd, show_console=False):
        self.cmd = cmd
        self.show_console = show_console
        self.proc = None
        self.stopped = False
        self._thread = None

    def start(self, on_line, on_progress, on_done):
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = process_flags(self.show_console)
            if self.show_console and os.name == "nt":
                # a visible console owns the output; there is nothing to pipe
                self.proc = subprocess.Popen(self.cmd, **kw)
            else:
                self.proc = subprocess.Popen(
                    self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                    errors="replace", bufsize=1, **kw)
        except OSError as exc:
            on_done(-1, "Could not start the converter: %s" % exc)
            return

        def pump():
            try:
                if self.proc.stdout is not None:
                    for raw in self.proc.stdout:
                        line = raw.rstrip("\r\n")
                        on_line(line)
                        pct = parse_progress(line)
                        ph = phase_of(line)
                        if pct is not None or ph is not None:
                            on_progress(pct, ph)
                rc = self.proc.wait()
            except Exception as exc:  # keep the window alive
                on_done(-1, "Lost the converter's output: %s" % exc)
                return
            if self.stopped:
                on_done(rc, "Stopped. Nothing was published (the converter "
                            "publishes the pack and its book together, only "
                            "when it finishes).")
            elif rc == 0:
                on_done(0, "Done.")
            else:
                on_done(rc, "The converter failed (exit code %d). See the "
                            "log above for the reason." % rc)

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.stopped = True
            self.proc.terminate()


# ------------------------------------------------------------ settings
def settings_path():
    return os.path.join(os.path.expanduser("~"), ".nanoquattwo",
                        "nq2-gui.json")


def load_settings(path=None):
    """Saved dialog settings; {} if none or unreadable. Lives in the user's
    home, never in the repository."""
    try:
        with open(path or settings_path(), "r", encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_settings(d, path=None):
    p = path or settings_path()
    try:
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2)
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def clean_threads(v):
    """A thread count clamped to 1..MAX_THREADS (default on junk)."""
    try:
        return max(1, min(MAX_THREADS, int(v)))
    except (TypeError, ValueError):
        return default_threads()
