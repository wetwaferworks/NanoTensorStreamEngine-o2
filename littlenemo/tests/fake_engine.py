#!/usr/bin/env python3
"""fake_engine.py - TEST FIXTURE ONLY, not part of the product and not
a model. Speaks PROTOCOL.md v2 plus the v3 extensions so every GUI
path can be exercised end to end without running a real engine:
streamed thinking (THINK), the answer (TOK), LAYER/TICK telemetry,
PREFILL, EXPAND, DTC, HS, MEM, MOE, STAT, HW, extended STATS, MEDIA /
FRAME attachments with validation (MEDIA_OK / MEDIA_ERR), STOP and
RESET. Its READY line carries fixture=1 and its replies say plainly
that they come from a fixture, so a screenshot can never pass a
fixture reply off as model output.

    python fake_engine.py --stdio [--model supernemo|littlenemo]
    python fake_engine.py --port PORT [...]

Options:
    --model M     supernemo: v3, media=image,video, MoE telemetry
                  littlenemo: text-only READY (no media key), like
                  the real Little Nemo engine
    --delay S     seconds between streamed tokens (default 0.03)
    --v2          emit only the v2 protocol (no v3 lines at all)
    --accel LANE  lane to report in HW accel= (default cpu)
    --think N     thinking tokens to stream (default: the whole trace)

Costs nothing: no model, no weights, a few sleeps.
"""
import os
import queue
import socket
import sys
import threading
import time
from urllib.parse import quote, unquote

MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_VIDEO_BYTES = 4 * 1024 * 1024 * 1024
MAX_FRAMES = 32
MAX_PENDING = 8
TYPES = ["mamba", "mlp", "attention"]


def esc(s):
    return s.replace("\\", "\\\\").replace("\n", "\\n").replace("\t", "\\t")


def unesc(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            out.append({"n": "\n", "t": "\t", "\\": "\\"}.get(s[i + 1], s[i + 1]))
            i += 2
        else:
            out.append(s[i])
            i += 1
    return "".join(out)


def kv(rest):
    d = {}
    for tok in rest.split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            d[k] = unquote(v)
    return d


def q(v):
    return quote(str(v), safe="")


class Fixture:
    def __init__(self, out, inbox, opts):
        self.out = out
        self.inbox = inbox
        self.o = opts
        self.super = opts["model"] == "supernemo"
        self.layers = 48 if self.super else 56
        self.pending = {}   # media id -> meta
        self.hs = {"pass": 0, "ordinal": 0, "hits": 0, "misses": 0,
                   "stall_ms": 0.0, "dtc": 0, "dtc_ms": 0.0}
        self.peak = 300.0
        self.deferred = []  # lines that arrived during a turn

    def w(self, line, ext=False):
        if ext and self.o["v2"]:
            return
        self.out.write(line + "\n")

    def flush(self):
        self.out.flush()

    def ready(self):
        cores = os.cpu_count() or 1
        if self.super and not self.o["v2"]:
            self.w(f"READY nemo-engine 3 model=SuperNemo-fixture "
                   f"weights=packed layers={self.layers} cores={cores} "
                   f"threads={max(1, cores - 1)} mode=max simd=1 "
                   f"media=image,video accel={self.o['accel']} moe=512 "
                   f"fixture=1")
        else:
            self.w(f"READY nemo-engine 2 model="
                   f"{'SuperNemo' if self.super else 'LittleNemo'}-fixture "
                   f"weights=raw layers={self.layers} cores={cores} "
                   f"threads={max(1, cores - 1)} mode=max simd=0 fixture=1")
        self.w(f"HW accel={self.o['accel']} note={q('reported by the test fixture')}",
               ext=True)
        self.flush()

    # -- attachments
    def media(self, rest):
        d = kv(rest)
        mid = d.get("id", "?")

        def err(reason):
            self.w(f"MEDIA_ERR id={q(mid)} reason={q(reason)}", ext=False)

        if not self.super or self.o["v2"]:
            return err("this engine is text-only")
        try:
            mid_i = int(mid)
            size = int(d.get("bytes", "x"))
        except ValueError:
            return err("id and bytes must be integers")
        typ = d.get("type")
        if typ not in ("image", "video"):
            return err("type must be image or video")
        if mid_i in self.pending:
            return err("duplicate media id")
        if len(self.pending) >= MAX_PENDING:
            return err(f"at most {MAX_PENDING} attachments per turn")
        limit = MAX_IMAGE_BYTES if typ == "image" else MAX_VIDEO_BYTES
        if not 0 < size <= limit:
            return err(f"bytes out of range (1..{limit})")
        path = d.get("path", "")
        if not path or not os.path.isfile(path):
            return err("file not found on the engine machine")
        if os.path.getsize(path) != size:
            return err("size on disk does not match bytes=")
        sent = int(d.get("sent_frames", 0) or 0)
        if sent > MAX_FRAMES:
            return err(f"at most {MAX_FRAMES} frames")
        meta = dict(d, id=mid_i, frames_seen=0, sent_frames=sent)
        self.pending[mid_i] = meta
        if sent == 0:
            self._accept(meta)

    def frame(self, rest):
        d = kv(rest)
        try:
            mid = int(d.get("id", "x"))
        except ValueError:
            return
        meta = self.pending.get(mid)
        if meta is None or meta.get("type") != "video":
            self.w(f"MEDIA_ERR id={mid} reason={q('FRAME for an unknown video')}")
            return
        if not os.path.isfile(d.get("path", "")):
            self.w(f"MEDIA_ERR id={mid} reason={q('frame file not found')}")
            self.pending.pop(mid, None)
            return
        meta["frames_seen"] += 1
        if meta["frames_seen"] == meta["sent_frames"]:
            self._accept(meta)

    def _accept(self, meta):
        toks = 256 if meta["type"] == "image" else 64 * max(1, meta["frames_seen"])
        meta["tokens"] = toks
        self.w(f"MEDIA_OK id={meta['id']} type={meta['type']} tokens={toks}")
        self.flush()

    # -- one turn
    def stopped(self):
        """Look at lines that arrived mid-generation: STOP ends the
        turn, QUIT / end of input end it and are handed back to the
        main loop, anything else waits until the turn is over."""
        while True:
            try:
                line = self.inbox.get_nowait()
            except queue.Empty:
                return False
            if line == "STOP":
                return True
            if line in ("QUIT", None):
                self.deferred.append(line)
                return True
            self.deferred.append(line)

    def turn(self, prompt):
        media = [m for m in self.pending.values() if "tokens" in m]
        self.pending.clear()
        mtoks = sum(m["tokens"] for m in media)
        ptoks = max(1, len(prompt.split()) + 4) + mtoks
        t0 = time.time()
        d = self.o["delay"]
        time.sleep(min(0.3, d * 4))
        pre_ms = (time.time() - t0) * 1000.0
        self.w(f"PREFILL tokens={ptoks} ms={pre_ms:.1f} "
               f"tps={ptoks / max(pre_ms, 1e-3) * 1000.0:.1f}", ext=True)
        names = ", ".join(f"{m.get('name', '?')} ({m['type']})" for m in media)
        trace = (f"The prompt has {len(prompt)} characters. "
                 + (f"It came with {len(media)} attachment(s): {names}. "
                    "A fixture cannot see pixels, so only the metadata is "
                    "available. " if media else "No attachments came with it. ")
                 + "Planning a short reply that says plainly this is a test "
                 "fixture and not a model. Checking that the thinking line "
                 "updates as each piece of this trace streams in. Ready to "
                 "answer.")
        think = trace.split(" ")
        if self.o["think"] is not None:
            think = think[:self.o["think"]]
        answer = ("This reply comes from fake_engine.py, a test fixture, "
                  "not from a model. Your message arrived with "
                  f"{len(prompt)} characters"
                  + (f" and {len(media)} attachment(s) the fixture accepted "
                     f"({mtoks} placeholder tokens)" if media else "")
                  + ". Every panel you see was driven by protocol lines.").split(" ")
        tok = 0
        t_first = None
        stop = False
        for mode, words, verb in (("thinking", think, "THINK"),
                                  ("answering", answer, "TOK")):
            for i, word in enumerate(words):
                if self.stopped():
                    stop = True
                    break
                tok += 1
                ts = time.time()
                layer = (tok * 7) % self.layers
                self.w(f"LAYER {layer} {TYPES[layer % 3]}")
                tname = f"layers.{layer}.{'mixer.in_proj' if layer % 3 == 0 else 'mlp.up_proj'}"
                self.w(f"EXPAND layer={layer} tensor={q(tname)} rows=4096 "
                       f"cols=2688 bytes={4096 * 2688 * 2} lane=cpu state=begin",
                       ext=True)
                self.w(f"DTC layer={layer} tensor={q(tname.replace('in_proj', 'out_proj').replace('up_proj', 'down_proj'))} "
                       f"rows=2688 cols=4096 k=256 block=8 active={180 + tok % 60} "
                       f"lane={self.o['accel']} state=begin", ext=True)
                if self.super:
                    ex = ",".join(str((tok * 37 + j * 101) % 512) for j in range(4))
                    self.w(f"MOE layer={layer} experts={q(ex)} k=4 of=512", ext=True)
                text = word + (" " if i < len(words) - 1 else "")
                self.w(f"{verb} {esc(text)}")
                if t_first is None:
                    t_first = time.time()
                time.sleep(d)
                self.w(f"EXPAND layer={layer} tensor={q(tname)} state=end", ext=True)
                self.hs["pass"] += 1
                self.hs["ordinal"] += self.layers
                hit = tok % 9 != 0
                self.hs["hits" if hit else "misses"] += 1
                self.hs["stall_ms"] += 0.0 if hit else 1.7
                self.hs["dtc"] += 1
                self.hs["dtc_ms"] += d * 300.0
                self.w("HS " + " ".join(
                    f"{k}={v:.1f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in self.hs.items()) + " staged=2", ext=True)
                ms = (time.time() - ts) * 1000.0
                self.w(f"TICK mode={mode} layer={layer} next={(layer + 1) % self.layers} "
                       f"active=1 staged=2 tok={tok} ms_tok={ms:.1f} gbs=0.0")
                if tok % 8 == 1:
                    self.peak += 3.0
                    self.w(f"MEM ram_mb={self.peak - 20:.0f} peak_mb={self.peak:.0f} "
                           f"vram_mb={166 if self.o['accel'] != 'cpu' else 0} "
                           f"vram_peak_mb={166 if self.o['accel'] != 'cpu' else 0}",
                           ext=True)
                    self.w(f"STAT fixture=1 kv_cache_tokens={ptoks + tok} "
                           f"rope_theta=10000", ext=True)
                self.flush()
            if stop:
                break
        total = (time.time() - t0) * 1000.0
        gen_ms = max(1e-3, (time.time() - (t_first or time.time())) * 1000.0)
        ext = ""
        if not self.o["v2"]:
            ext = (f" prompt_tokens={ptoks} prompt_ms={pre_ms:.1f} "
                   f"ttft_ms={((t_first or time.time()) - t0) * 1000.0:.1f} "
                   f"gen_tps={max(0, tok - 1) / gen_ms * 1000.0:.2f} "
                   f"stopped={1 if stop else 0} fixture=1")
        self.w(f"STATS tokens={tok} ms={total:.1f} "
               f"tps={tok / max(total, 1e-3) * 1000.0:.3f} "
               f"ms_per_tok={total / max(tok, 1):.1f} gbs_avg=0.0 "
               f"peak_mb={self.peak:.1f}{ext}")
        self.w("DONE")
        self.flush()

    def serve(self):
        self.ready()
        while True:
            line = (self.deferred.pop(0) if self.deferred
                    else self.inbox.get())
            if line is None or line == "QUIT":
                return
            if line.startswith("PROMPT"):
                self.turn(unesc(line[7:]))
            elif line.startswith("MEDIA "):
                self.media(line[6:])
                self.flush()
            elif line.startswith("FRAME "):
                self.frame(line[6:])
                self.flush()
            elif line == "RESET":
                self.pending.clear()
            elif line == "STOP":
                pass  # nothing running


def reader(f, inbox):
    try:
        for line in f:
            inbox.put(line.rstrip("\r\n"))
    finally:
        inbox.put(None)


def main(argv):
    opts = {"model": "supernemo", "delay": 0.03, "v2": False,
            "accel": "cpu", "think": None}
    port = None
    stdio = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--stdio":
            stdio = True
        elif a == "--port":
            port = int(argv[i + 1]); i += 1
        elif a == "--model":
            opts["model"] = argv[i + 1]; i += 1
        elif a == "--delay":
            opts["delay"] = float(argv[i + 1]); i += 1
        elif a == "--v2":
            opts["v2"] = True
        elif a == "--accel":
            opts["accel"] = argv[i + 1]; i += 1
        elif a == "--think":
            opts["think"] = int(argv[i + 1]); i += 1
        elif a.isdigit() and port is None:
            port = int(a)  # mock_engine.py style: fake_engine.py PORT
        i += 1
    if opts["model"] not in ("supernemo", "littlenemo"):
        sys.stderr.write("--model must be supernemo or littlenemo\n")
        return 2
    inbox = queue.Queue()
    if stdio or port is None:
        sys.stdout.reconfigure(encoding="utf-8", newline="\n")
        sys.stdin.reconfigure(encoding="utf-8")
        threading.Thread(target=reader, args=(sys.stdin, inbox),
                         daemon=True).start()
        Fixture(sys.stdout, inbox, opts).serve()
        return 0
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    conn, _ = srv.accept()
    with conn:
        f = conn.makefile("rw", encoding="utf-8", newline="\n")
        threading.Thread(target=reader, args=(f, inbox), daemon=True).start()
        Fixture(f, inbox, opts).serve()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
