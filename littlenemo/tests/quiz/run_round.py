#!/usr/bin/env python3
"""Run the Little Nemo verifiable round: 50 short answers and 10 essays.

One engine process per phase (the engine's --max-new is set at launch),
RESET between items, greedy, chat template on, thinking off. Each item
records prefill and decode tokens/s from the engine's PREFILL and STATS
lines, its wall time, and the engine's peak RAM (MEM). A side thread
samples nvidia-smi once a second for the whole-card VRAM peak and GPU
load, so contention from other programs shows up in the record.

    python littlenemo/tests/quiz/run_round.py \\
        --engine path/to/nemo-engine-cuda.exe \\
        --checkpoint /path/to/nemotron-nano-9b-v2 \\
        --cuda-pack /path/to/the.nq2 --out results.json

No checkpoint, pack or machine path is stored in this file.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from grade import answer_text, grade_question, score_grades  # noqa: E402
from grade_essay import grade_essay, score_essays  # noqa: E402


def kv(line: str) -> dict:
    out = {}
    for part in line.split()[1:]:
        if "=" in part:
            k, v = part.split("=", 1)
            try:
                out[k] = float(v) if any(c in v for c in ".eE") else int(v)
            except ValueError:
                out[k] = v
    return out


class GpuSampler:
    """Polls nvidia-smi; one writer thread, the reader only looks after join."""

    def __init__(self, period: float = 1.0):
        self.period = period
        self.samples = []
        self._stop = False
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        q = ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
        while not self._stop:
            try:
                r = subprocess.run(q, capture_output=True, text=True, timeout=10)
                a = r.stdout.strip().splitlines()[0].split(",")
                self.samples.append((time.perf_counter(), int(a[0]), int(a[1])))
            except Exception:
                pass
            time.sleep(self.period)

    def start(self):
        self._t.start()
        return self

    def stop(self):
        self._stop = True
        self._t.join(timeout=15)

    def window(self, t0: float, t1: float) -> dict:
        s = [x for x in self.samples if t0 <= x[0] <= t1]
        if not s:
            return {}
        return {"vram_used_peak_mib": max(x[1] for x in s),
                "gpu_util_mean": round(sum(x[2] for x in s) / len(s), 1),
                "samples": len(s)}


def launch(args, max_new: int):
    if args.style == "supernemo":
        # sn-engine takes named flags (--pack, --config, ...) in --engine-args;
        # chat is on and thinking off unless --think is passed.
        cmd = [args.engine, "--max-new", str(max_new)] + args.engine_args
    else:
        cmd = [args.engine, args.checkpoint, args.tokenizer, "--chat", "--max-new", str(max_new)]
        if args.cuda_pack:
            cmd += ["--cuda-pack", args.cuda_pack]
        cmd += args.engine_args
    flags = 0
    if os.name == "nt" and args.priority == "belownormal":
        flags = 0x00004000  # BELOW_NORMAL_PRIORITY_CLASS
    err = open(args.stderr_log, "a", encoding="utf-8") if args.stderr_log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err,
                            text=True, encoding="utf-8", errors="replace", bufsize=1,
                            creationflags=flags)
    ready = ""
    while not ready:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited before READY")
        line = line.rstrip("\r\n")
        if line.startswith("READY"):
            ready = line
        elif line.startswith("ERROR"):
            raise RuntimeError(line)
    hw = ""
    if args.cuda_pack and args.style == "nemo":
        # The CUDA engine prints its HW lane line right after READY.
        line = proc.stdout.readline().rstrip("\r\n")
        if line.startswith("HW "):
            hw = line
    return proc, cmd, ready, hw


def turn(proc, prompt: str) -> dict:
    proc.stdin.write("PROMPT " + prompt.replace("\n", "\\n") + "\n")
    proc.stdin.flush()
    pieces, prefill, stats, mem, error = [], {}, {}, {}, ""
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited mid-turn")
        line = line.rstrip("\r\n")
        if line.startswith("TOK "):
            pieces.append(line[4:])
        elif line.startswith("PREFILL "):
            prefill = kv(line)
        elif line.startswith("STATS "):
            stats = kv(line)
        elif line.startswith("MEM "):
            mem = kv(line)
        elif line.startswith("ERROR"):
            error = line
            break
        elif line == "DONE":
            break
    return {"text": answer_text(pieces), "prefill": prefill, "stats": stats, "mem": mem, "error": error}


def profile(t: dict, wall: float) -> dict:
    s, p = t["stats"], t["prefill"]
    return {
        "prompt_tokens": p.get("tokens", s.get("prompt_tokens")),
        "prefill_ms": p.get("ms", s.get("prompt_ms")),
        "prefill_tps": p.get("tps", s.get("prompt_tps")),
        "gen_tokens": s.get("tokens"),
        "gen_ms": s.get("ms"),  # whole turn, prefill included
        "decode_tps": s.get("gen_tps", s.get("tps")),
        "stopped": s.get("stopped"),
        "dtc_steps": s.get("dtc_steps"),
        "expand_steps": s.get("expand_steps"),
        "ram_peak_mb": t["mem"].get("peak_mb", s.get("peak_mb")),
        "wall_s": round(wall, 3),
    }


def gate(args) -> tuple[int, str]:
    """Run the optional --gate-cmd. Exit 0 means go; any other code means
    another job holds the machine, so the engine is closed and we wait."""
    if not args.gate_cmd:
        return 0, ""
    r = subprocess.run(args.gate_cmd, shell=True, capture_output=True, text=True)
    lines = (r.stdout or "").strip().splitlines()
    return r.returncode, (lines[-1] if lines else "")


def run_phase(args, items, max_new, sampler, grade_fn, label):
    t_start = time.perf_counter()
    proc, cmd, ready, hw = None, None, "", ""
    load_s = 0.0
    launches = 0
    pauses = []
    rows = []

    def close(p):
        if p is None:
            return
        try:
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=60)
        except Exception:
            pass
        if p.poll() is None:
            p.kill()

    try:
        for it in items:
            rc, why = gate(args)
            if rc != 0:
                # Free the GPU and cores for the other job, then wait it out.
                close(proc)
                proc = None
                p0 = time.time()
                print("%s%-2d paused: %s" % (label, it["id"], why), flush=True)
                while rc != 0:
                    time.sleep(args.gate_poll)
                    rc, why2 = gate(args)
                pauses.append({"before_item": it["id"], "reason": why,
                               "from": time.strftime("%H:%M:%S", time.localtime(p0)),
                               "to": time.strftime("%H:%M:%S"), "s": round(time.time() - p0, 1)})
            if proc is None:
                t0 = time.perf_counter()
                proc, cmd, ready, hw = launch(args, max_new)
                load_s += time.perf_counter() - t0
                launches += 1
                print(ready, flush=True)
            elif rows:
                proc.stdin.write("RESET\n")
                proc.stdin.flush()
                if args.style == "supernemo":
                    # sn-engine answers RESET with DONE; nemo-engine sends nothing.
                    while True:
                        ack = proc.stdout.readline()
                        if ack == "":
                            raise RuntimeError("engine exited after RESET")
                        if ack.rstrip("\r\n") == "DONE":
                            break
            g_before = why
            t0 = time.perf_counter()
            t = turn(proc, it["prompt"])
            t1 = time.perf_counter()
            g_after = gate(args) if args.gate_cmd else (0, "")
            row = {"id": it["id"], "prompt": it["prompt"], "text": t["text"], "error": t["error"]}
            row.update(grade_fn(it, t["text"]))
            row["profile"] = profile(t, t1 - t0)
            row["profile"].update(sampler.window(t0, t1))
            if args.gate_cmd:
                row["gate_before"] = g_before
                row["gate_after"] = g_after[1]
                row["overlap"] = g_after[0] != 0
            rows.append(row)
            pr = row["profile"]
            print("%s%-2d %-5s prefill=%s tps  decode=%s tps  %s" % (
                label, it["id"], row["grade"], pr["prefill_tps"], pr["decode_tps"],
                (row.get("first_line") or row.get("topic") or "")[:60]), flush=True)
        close(proc)
        proc = None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
    t_end = time.perf_counter()
    return {"command": cmd, "ready": ready, "hw": hw, "max_new": max_new,
            "load_s": round(load_s, 3), "launches": launches, "pauses": pauses,
            "wall_s": round(t_end - t_start, 3),
            "gpu": sampler.window(t_start, t_end), "items": rows}


def summarize(rows) -> dict:
    def col(k):
        return [r["profile"][k] for r in rows if isinstance(r["profile"].get(k), (int, float))]
    out = {}
    for k in ("prefill_tps", "decode_tps"):
        v = sorted(col(k))
        if v:
            out[k] = {"mean": round(sum(v) / len(v), 3), "min": v[0], "median": v[len(v) // 2], "max": v[-1]}
    gt, gm = sum(col("gen_tokens")), sum(col("gen_ms"))
    pt, pm = sum(col("prompt_tokens")), sum(col("prefill_ms"))
    # STATS ms covers the whole turn, prompt included, so take prefill out.
    if gm - pm > 0:
        out["decode_tps_pooled"] = round(gt * 1000.0 / (gm - pm), 3)
    if pm:
        out["prefill_tps_pooled"] = round(pt * 1000.0 / pm, 3)
    out["gen_tokens"] = gt
    out["prompt_tokens"] = pt
    out["ram_peak_mb"] = max(col("ram_peak_mb"), default=None)
    out["vram_card_peak_mib"] = max(col("vram_used_peak_mib"), default=None)
    out["expand_steps"] = sum(col("expand_steps"))
    out["dtc_steps"] = sum(col("dtc_steps"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--checkpoint", default=os.environ.get("NEMO_CKPT"))
    ap.add_argument("--tokenizer", default=os.environ.get("NEMO_TOKENIZER"))
    ap.add_argument("--cuda-pack", default=os.environ.get("NEMO_CUDA_PACK"))
    ap.add_argument("--questions", type=Path, default=HERE / "questions50.json")
    ap.add_argument("--essays", type=Path, default=HERE / "essays.json")
    ap.add_argument("--q-max-new", type=int, default=16)
    ap.add_argument("--essay-max-new", type=int, default=320)
    ap.add_argument("--skip-questions", action="store_true")
    ap.add_argument("--skip-essays", action="store_true")
    ap.add_argument("--priority", choices=["normal", "belownormal"], default="belownormal")
    ap.add_argument("--stderr-log", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--style", choices=["nemo", "supernemo"], default="nemo",
                    help="nemo: engine CKPT TOKENIZER --chat; supernemo: sn-engine with named flags in --engine-args")
    ap.add_argument("--ids", default="", help="comma list of question ids to run (default all)")
    ap.add_argument("--essay-ids", default="", help="comma list of essay ids to run (default all)")
    ap.add_argument("--gate-cmd", default="", help="command run before and after each item; nonzero means another job holds the machine")
    ap.add_argument("--gate-poll", type=float, default=60.0)
    ap.add_argument("--engine-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args(argv)
    if args.style == "nemo" and not args.checkpoint:
        ap.error("need --checkpoint or NEMO_CKPT")
    if args.checkpoint and not args.tokenizer:
        args.tokenizer = str(Path(args.checkpoint) / "tokenizer.json")
    if args.engine_args and args.engine_args[0] == "--":
        args.engine_args = args.engine_args[1:]

    sampler = GpuSampler().start()
    payload = {"started": time.strftime("%Y-%m-%d %H:%M:%S %z")}
    t0 = time.perf_counter()
    try:
        if not args.skip_questions:
            qz = json.loads(args.questions.read_text(encoding="utf-8"))
            qsel = qz["questions"]
            if args.ids:
                want = {int(x) for x in args.ids.split(",")}
                qsel = [q for q in qsel if q["id"] in want]
            ph = run_phase(args, qsel, args.q_max_new, sampler,
                           lambda q, txt: grade_question(q, txt), "Q")
            ph["score"] = score_grades(ph["items"])
            ph["summary"] = summarize(ph["items"])
            payload["questions"] = ph
            print("QUESTIONS", ph["score"]["score"], json.dumps(ph["summary"]), flush=True)
        if not args.skip_essays:
            ez = json.loads(args.essays.read_text(encoding="utf-8"))
            pf = ez.get("pass_fraction", 0.7)
            esel = ez["essays"]
            if args.essay_ids:
                want = {int(x) for x in args.essay_ids.split(",")}
                esel = [e for e in esel if e["id"] in want]
            ph = run_phase(args, esel, args.essay_max_new, sampler,
                           lambda e, txt: grade_essay(e, txt, pf), "E")
            ph["score"] = score_essays(ph["items"])
            ph["summary"] = summarize(ph["items"])
            payload["essays"] = ph
            print("ESSAYS", ph["score"]["score"], json.dumps(ph["summary"]), flush=True)
    finally:
        sampler.stop()
    payload["wall_s"] = round(time.perf_counter() - t0, 3)
    payload["gpu_whole_run"] = sampler.window(0, float("inf"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", args.out, "wall=%.1fs" % payload["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
