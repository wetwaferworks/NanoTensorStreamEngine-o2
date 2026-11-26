#!/usr/bin/env python3
"""nemo_stats.py - the stats panel's data side and the thinking line's
headline (no Tk, headless-testable: tests/test_gui_stats.py).

StatsModel folds protocol events (v2 and the v3 extensions) into the
rows the right-hand stats panel shows: live throughput (prompt and
generation tok/s, time to first token), hardware, what is expanding
and what is in dense tensor compute right now, Tensor Handshake
counters, memory, the last turn's STATS, the READY receipt, and any
other statistic the engine reports. Every number shown is one the
engine measured or the GUI measured itself (labeled "gui"), or one the
GUI probed from this machine (labeled "probe"); nothing is invented,
and a value nobody reported shows as a dash.

probe_hardware() reads this machine's specs once, on a worker thread.
"""

import json
import os
import platform
import re
import subprocess
import sys
import time

DASH = "\u2014"


# ------------------------------------------------------- thinking line ---

def thinking_headline(text, limit=96):
    """What the model is thinking about right now, as one line: the
    latest sentence of the reasoning trace (joined with the previous
    one when the latest is only a fragment), tail-trimmed to `limit`
    characters on a word boundary."""
    s = " ".join((text or "").split())
    if not s:
        return ""
    parts = [p for p in re.split(r"(?<=[.!?;:])\s+", s) if p]
    head = parts[-1]
    if len(head) < 28 and len(parts) > 1:
        head = parts[-2] + " " + head
    if len(head) <= limit:
        return head
    tail = head[-(limit - 1):]
    cut = tail.find(" ")
    if 0 <= cut < limit // 3:
        tail = tail[cut + 1:]
    return "\u2026" + tail


# ------------------------------------------------------------- hardware ---

def _run(cmd, timeout=6):
    try:
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000  # CREATE_NO_WINDOW
        r = subprocess.run(cmd, capture_output=True, text=True,
                           timeout=timeout, **kw)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


def _is_integrated(name):
    n = name.lower()
    return any(k in n for k in ("intel", "uhd", "iris", "radeon(tm) graphics",
                                "radeon graphics", "vega", "apple"))


def ram_total_mb():
    if os.name == "nt":
        try:
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong),
                            ("total", ctypes.c_ulonglong),
                            ("avail", ctypes.c_ulonglong),
                            ("tp", ctypes.c_ulonglong),
                            ("ap", ctypes.c_ulonglong),
                            ("tv", ctypes.c_ulonglong),
                            ("av", ctypes.c_ulonglong),
                            ("ae", ctypes.c_ulonglong)]
            m = MS()
            m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.total // (1024 * 1024), m.avail // (1024 * 1024)
        except Exception:
            return None, None
    try:
        with open("/proc/meminfo") as f:
            info = dict(l.split(":", 1) for l in f if ":" in l)
        tot = int(info["MemTotal"].split()[0]) // 1024
        av = int(info.get("MemAvailable", "0 kB").split()[0]) // 1024
        return tot, av
    except Exception:
        return None, None


def nvidia_smi():
    """[{'name','vram_mb','vram_used_mb','driver'}] or []."""
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,"
                "driver_version", "--format=csv,noheader,nounits"])
    gpus = []
    for line in out.strip().splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) >= 4:
            try:
                gpus.append({"name": p[0], "vram_mb": int(float(p[1])),
                             "vram_used_mb": int(float(p[2])),
                             "driver": p[3]})
            except ValueError:
                pass
    return gpus


def probe_hardware():
    """This machine's specs -> dict (keys may be missing). Slow-ish
    (spawns PowerShell / nvidia-smi once): call from a worker thread."""
    hw = {"os": f"{platform.system()} {platform.release()}",
          "threads": os.cpu_count()}
    tot, _av = ram_total_mb()
    if tot:
        hw["ram_mb"] = tot
    adapters = []
    if os.name == "nt":
        try:
            import winreg

            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                               r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            hw["cpu"] = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except Exception:
            pass
        ps = ("$p=Get-CimInstance Win32_Processor | Select-Object -First 1 "
              "NumberOfCores,NumberOfLogicalProcessors;"
              "$v=@(Get-CimInstance Win32_VideoController | "
              "Select-Object Name,AdapterRAM,DriverVersion);"
              "@{p=$p;v=$v} | ConvertTo-Json -Depth 3 -Compress")
        out = _run(["powershell", "-NoProfile", "-Command", ps], timeout=15)
        try:
            data = json.loads(out)
            p = data.get("p") or {}
            if p.get("NumberOfCores"):
                hw["cores"] = int(p["NumberOfCores"])
            if p.get("NumberOfLogicalProcessors"):
                hw["threads"] = int(p["NumberOfLogicalProcessors"])
            v = data.get("v") or []
            if isinstance(v, dict):
                v = [v]
            for a in v:
                adapters.append({"name": (a.get("Name") or "").strip(),
                                 "vram_mb": int(a.get("AdapterRAM") or 0)
                                 // (1024 * 1024),
                                 "driver": a.get("DriverVersion") or ""})
        except Exception:
            pass
    else:
        try:
            with open("/proc/cpuinfo") as f:
                txt = f.read()
            m = re.search(r"model name\s*:\s*(.+)", txt)
            if m:
                hw["cpu"] = m.group(1).strip()
            cores = {(a, b) for a, b in re.findall(
                r"physical id\s*:\s*(\d+).*?core id\s*:\s*(\d+)", txt, re.S)}
            if cores:
                hw["cores"] = len(cores)
        except Exception:
            pass
    if "cpu" not in hw and platform.processor():
        hw["cpu"] = platform.processor()
    nv = nvidia_smi()
    if nv:
        g = nv[0]
        hw["gpu"], hw["vram_mb"], hw["driver"] = (g["name"], g["vram_mb"],
                                                  g["driver"])
        hw["vram_used_mb"] = g["vram_used_mb"]
        hw["cuda"] = "nvidia-smi present"
    for a in adapters:
        if not a["name"]:
            continue
        if _is_integrated(a["name"]):
            hw.setdefault("igpu", a["name"])
        elif "gpu" not in hw:
            hw["gpu"], hw["driver"] = a["name"], a["driver"]
            if a["vram_mb"]:
                hw["vram_mb"] = a["vram_mb"]
    return hw


def process_rss_mb(pid):
    """Resident memory of a local process in MB, or None."""
    if not pid:
        return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD),
                            ("PageFaultCount", wintypes.DWORD),
                            ("PeakWorkingSetSize", ctypes.c_size_t),
                            ("WorkingSetSize", ctypes.c_size_t),
                            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                            ("PagefileUsage", ctypes.c_size_t),
                            ("PeakPagefileUsage", ctypes.c_size_t)]
            k32 = ctypes.windll.kernel32
            k32.OpenProcess.restype = wintypes.HANDLE
            h = k32.OpenProcess(0x1000 | 0x0010, False, int(pid))
            if not h:
                return None
            try:
                c = PMC()
                c.cb = ctypes.sizeof(PMC)
                if not k32.K32GetProcessMemoryInfo(h, ctypes.byref(c), c.cb):
                    return None
                return c.WorkingSetSize / (1024 * 1024)
            finally:
                k32.CloseHandle(h)
        except Exception:
            return None
    try:
        with open(f"/proc/{int(pid)}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        return None
    return None


# ---------------------------------------------------------- stats model ---

def _f(v, fmt="{:.1f}", unit=""):
    if v is None or v == "":
        return DASH
    try:
        return fmt.format(float(v)) + unit
    except (TypeError, ValueError):
        return str(v)


def _mb(v):
    if v is None or v == "":
        return DASH
    v = float(v)
    return f"{v / 1024:.2f} GB" if v >= 1024 else f"{v:.0f} MB"


_LANES = {"cpu": "CPU", "cuda": "CUDA", "opencl": "OpenCL",
          "opencl-igpu": "OpenCL (integrated GPU)",
          "opencl-nv": "OpenCL (NVIDIA, no CUDA)"}


class StatsModel:
    """Folds protocol events into stats-panel rows. Feed it every
    event with on_event(kind, payload, now); call turn_begin() when a
    turn is sent. rows() -> [(section, [(label, value), ...]), ...]."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.probe = {}
        self.reset_engine()

    def reset_engine(self):
        self.ready = {}
        self.hw = {}
        self.layer = None
        self.layer_type = None
        self.layer_types = {}
        self.tick = {}
        self.expand = None
        self.expand_count = 0
        self.dtc = None
        self.dtc_count = 0
        self.moe = None
        self.hs = {}
        self.mem = {}
        self.other = {}
        self.last_stats = {}
        self.prefill = {}
        self.engine_rss_mb = None
        self.vram_sample = None
        self.turns = 0
        self._new_turn()

    def _new_turn(self):
        self.t_send = None
        self.t_first = None
        self.t_last = None
        self.t_done = None
        self.think_tokens = 0
        self.answer_tokens = 0
        self.tokens = 0
        self.media = []

    def turn_begin(self, now=None):
        self._new_turn()
        self.prefill = {}
        self.t_send = self.clock() if now is None else now
        self.turns += 1

    # -- events
    def on_event(self, kind, payload, now=None):
        now = self.clock() if now is None else now
        if kind == "ready":
            self.reset_engine()
            self.ready = dict(payload or {})
        elif kind in ("think", "tok"):
            if self.t_first is None and self.t_send is not None:
                self.t_first = now
            self.t_last = now
        elif kind == "layer":
            self.layer = payload["idx"]
            self.layer_type = payload["type"]
            self.layer_types[payload["idx"]] = payload["type"]
        elif kind == "tick":
            self.tick = dict(payload)
            self.layer = payload.get("layer", self.layer)
            mode = payload.get("mode")
            if mode == "thinking":
                self.think_tokens += 1
            elif mode == "answering":
                self.answer_tokens += 1
            self.tokens = payload.get("tok", self.tokens + 1)
        elif kind == "hw":
            self.hw.update(payload)
        elif kind == "prefill":
            self.prefill = dict(payload)
        elif kind == "expand":
            if payload.get("state") == "end":
                self.expand = dict(payload, state="done")
            else:
                self.expand = dict(payload)
                self.expand_count += 1
        elif kind == "dtc":
            if payload.get("state") == "end":
                self.dtc = dict(payload, state="done")
            else:
                self.dtc = dict(payload)
                self.dtc_count += 1
        elif kind == "moe":
            self.moe = dict(payload)
        elif kind == "hs":
            self.hs.update(payload)
        elif kind == "mem":
            self.mem.update(payload)
        elif kind == "stat":
            self.other.update(payload)
        elif kind in ("media_ok", "media_err"):
            self.media.append((kind, dict(payload)))
        elif kind == "stats":
            self.last_stats = dict(payload)
            self.t_done = now
        elif kind in ("done", "error"):
            if self.t_done is None:
                self.t_done = now

    # -- derived numbers
    def ttft_ms(self):
        """(value, source): engine-reported ttft_ms, else measured by
        the GUI from send to the first THINK/TOK."""
        s = self.last_stats if self.t_done else {}
        if "ttft_ms" in s:
            return float(s["ttft_ms"]), "engine"
        if self.t_send is not None and self.t_first is not None:
            return (self.t_first - self.t_send) * 1000.0, "gui"
        return None, None

    def prompt_tps(self):
        """Prompt processing rate, engine-reported only (PREFILL, or
        prompt_tps in the turn's STATS)."""
        if "tps" in self.prefill:
            return float(self.prefill["tps"]), "engine"
        if self.t_done and "prompt_tps" in self.last_stats:
            return float(self.last_stats["prompt_tps"]), "engine"
        return None, None

    def gen_tps(self, now=None):
        """Live generation rate: tokens after the first one over the
        time since the first one (GUI-measured), or the engine's gen
        rate once the turn's STATS arrive."""
        if self.t_done and self.last_stats:
            for k in ("gen_tps", "tps"):
                if k in self.last_stats:
                    return float(self.last_stats[k]), "engine"
        now = self.clock() if now is None else now
        n = self.think_tokens + self.answer_tokens
        if self.t_first is None or n < 2:
            return None, None
        span = (self.t_last or now) - self.t_first
        if span <= 0:
            return None, None
        return (n - 1) / span, "gui"

    def lane(self):
        a = self.hw.get("accel") or self.ready.get("accel")
        if a:
            return _LANES.get(str(a).lower(), str(a)), "engine"
        simd = self.ready.get("simd")
        if self.ready:
            return ("CPU" + (" + AVX2" if simd == "1" else " (scalar)"),
                    "READY")
        return None, None

    # -- rows
    def rows(self, now=None):
        now = self.clock() if now is None else now

        def src(v, s):
            return v if s in (None, "engine") else f"{v}  ({s})"

        t, ts = self.ttft_ms()
        p, ps = self.prompt_tps()
        g, gs = self.gen_tps(now)
        tick = self.tick
        ms_tok = tick.get("ms_tok")
        elapsed = None
        if self.t_send is not None:
            elapsed = (self.t_done or now) - self.t_send
        thr = [
            ("Time to first token", src(_f(t, "{:.0f}", " ms"), ts)),
            ("Prompt", src(_f(p, "{:.1f}", " tok/s"), ps)
             + (f"  · {self.prefill.get('tokens')} tok"
                if "tokens" in self.prefill else "")),
            ("Generation", src(_f(g, "{:.2f}", " tok/s"), gs)),
            ("Last token", _f(ms_tok, "{:.0f}", " ms")
             + (f"  · {1000.0 / ms_tok:.2f} tok/s"
                if isinstance(ms_tok, (int, float)) and ms_tok > 0 else "")),
            ("Tokens (think / answer)",
             f"{self.think_tokens} / {self.answer_tokens}"
             if self.t_send is not None else DASH),
            ("Memory throughput", _f(tick.get("gbs"), "{:.1f}", " GB/s est")),
            ("Elapsed", _f(elapsed, "{:.1f}", " s")),
        ]

        hw = dict(self.probe)
        engine_keys = set(self.hw)
        hw.update(self.hw)

        def hsrc(key, text):
            if text == DASH:
                return text
            return text if key in engine_keys else f"{text}  (probe)"

        cores = hw.get("cores")
        threads = hw.get("threads")
        lane, lane_src = self.lane()
        hard = [
            ("CPU", hsrc("cpu", str(hw.get("cpu") or DASH))),
            ("Cores / threads",
             hsrc("cores", f"{cores or DASH} / {threads or DASH}")),
            ("RAM", hsrc("ram_mb", _mb(hw.get("ram_mb")))),
            ("GPU", hsrc("gpu", str(hw.get("gpu") or DASH))),
            ("VRAM", hsrc("vram_mb", _mb(hw.get("vram_mb")))),
            ("Driver", hsrc("driver", str(hw.get("driver") or DASH))),
            ("Integrated GPU", hsrc("igpu", str(hw.get("igpu") or DASH))),
            ("Accel lane", src(lane or DASH, lane_src if lane_src != "engine"
                               else None)),
            ("Engine lanes",
             f"{self.ready.get('threads', DASH)} of "
             f"{self.ready.get('cores', DASH)} cores · mode "
             f"{self.ready.get('mode', DASH)}" if self.ready else DASH),
        ]

        lyr = self.layer
        nlayers = self.ready.get("layers", DASH)
        ex = self.expand
        dt = self.dtc
        now_rows = [
            ("Mode", "idle" if (self.t_done or self.t_send is None)
             else str(tick.get("mode") or "prefill")),
            ("Layer", f"{lyr} / {nlayers}"
             + (f"  ({self.layer_types.get(lyr, self.layer_type)})"
                if lyr is not None and self.layer_types.get(lyr) else "")
             if lyr is not None else DASH),
            ("Next / staged", f"{tick.get('next', DASH)} / "
             f"{tick.get('staged', DASH)}  · active {tick.get('active', DASH)}"
             if tick else DASH),
            ("Expanding", _describe(ex, ("tensor", "layer", "rows", "cols",
                                         "bytes", "lane"))),
            ("Expansions this session", str(self.expand_count)),
            ("In DTC", _describe(dt, ("tensor", "layer", "rows", "cols", "k",
                                      "block", "active", "lane"))),
            ("DTC ops this session", str(self.dtc_count)),
        ]
        if self.moe:
            now_rows.append(("MoE experts",
                             f"layer {self.moe.get('layer', DASH)}: "
                             f"{self.moe.get('experts', DASH)}"
                             + (f"  (top-{self.moe['k']} of "
                                f"{self.moe.get('of', DASH)})"
                                if "k" in self.moe else "")))

        hs = self.hs
        hits, miss = hs.get("hits"), hs.get("misses")
        rate = DASH
        if isinstance(hits, (int, float)) and isinstance(miss, (int, float)) \
                and hits + miss > 0:
            rate = f"{100.0 * hits / (hits + miss):.1f}%"
        hand = [
            ("Pass / ordinal", f"{hs.get('pass', DASH)} / "
             f"{hs.get('ordinal', DASH)}" if hs else DASH),
            ("Prefetch hits / misses", f"{hits if hits is not None else DASH}"
             f" / {miss if miss is not None else DASH}  · hit rate {rate}"
             if hs else DASH),
            ("Stall time", _f(hs.get("stall_ms"), "{:.1f}", " ms")),
            ("DTC requests / time", f"{hs.get('dtc', DASH)} / "
             f"{_f(hs.get('dtc_ms'), '{:.1f}', ' ms')}" if hs else DASH),
            ("Buffers staged", str(hs.get("staged", DASH)) if hs else DASH),
        ]
        for k in sorted(set(hs) - {"pass", "ordinal", "hits", "misses",
                                   "stall_ms", "dtc", "dtc_ms", "staged"}):
            hand.append((k, str(hs[k])))

        mem = self.mem
        memory = [
            ("RAM in use", _mb(mem["ram_mb"]) if "ram_mb" in mem else
             (f"{_mb(self.engine_rss_mb)}  (probe, engine process)"
              if self.engine_rss_mb else DASH)),
            ("RAM peak", _mb(mem.get("peak_mb",
                                     self.last_stats.get("peak_mb")))),
            ("VRAM in use", _mb(mem["vram_mb"]) if "vram_mb" in mem else
             (f"{_mb(self.vram_sample)}  (probe, whole GPU)"
              if self.vram_sample is not None else DASH)),
            ("VRAM peak", _mb(mem.get("vram_peak_mb"))),
        ]
        for k in sorted(set(mem) - {"ram_mb", "peak_mb", "vram_mb",
                                    "vram_peak_mb"}):
            memory.append((k, str(mem[k])))

        last = [(k, _fmt_any(v)) for k, v in self.last_stats.items()]
        ready = [(k, str(v)) for k, v in self.ready.items()]
        other = [(k, _fmt_any(v)) for k, v in sorted(self.other.items())]
        media = [(f"#{p.get('id', '?')}",
                  (f"accepted · {p.get('tokens', DASH)} tokens"
                   if k == "media_ok" else
                   f"refused · {p.get('reason', '')}"))
                 for k, p in self.media]
        out = [("Throughput", thr), ("Hardware", hard), ("Now", now_rows),
               ("Tensor Handshake", hand), ("Memory", memory)]
        if media:
            out.append(("Attachments this turn", media))
        out.append(("Last turn (STATS)", last or [("", DASH)]))
        if other:
            out.append(("Other engine stats", other))
        out.append(("Engine READY", ready or [("", "not connected")]))
        return out


def _fmt_any(v):
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def _describe(d, keys):
    if not d:
        return DASH
    bits = []
    for k in keys:
        if k in d and d[k] not in ("", None):
            v = d[k]
            if k == "bytes":
                v = f"{float(v) / (1024 * 1024):.2f} MB"
            bits.append(str(v) if k == "tensor" else f"{k} {v}")
    if d.get("state") == "done":
        bits.append("(finished)")
    return "  ".join(bits) if bits else DASH
