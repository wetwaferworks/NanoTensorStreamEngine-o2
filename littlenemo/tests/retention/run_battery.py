#!/usr/bin/env python3
"""Run the NanoTensor retention battery on the Little Nemo CUDA engine.

Greedy, chat template on, thinking off, RESET between items, one engine
process per token cap. With NEMO_LOGIT_TOPK set (the runner sets it),
the engine prints one LOGIT line per generated token: chosen-token
log-probability, margin to the runner-up, stop-set log-probability and
rank, and the top tokens. The runner keeps them per item and derives
the termination numbers: whether the turn ended on its own, and the
stop-set probability at the end of the first answer line (where a
clean model ends its turn).

Every item records prefill and decode tokens/s, wall time, engine RAM
peak, whole-card VRAM peak and the lane (cuda or cpu) from the engine's
own lines, plus nvidia-smi samples, as JD asked for every test.

    python littlenemo/tests/retention/run_battery.py --engine ENGINE \\
        --checkpoint CKPT --cuda-pack PACK --arm packed --out results.json \\
        [--sets quiz,para,...] [--essays] [--env NEMO_GPU_HEAD=bf16]

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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
QUIZ = HERE.parent / "quiz"
for p in (HERE, QUIZ):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
from grade import answer_text  # noqa: E402
from grade_essay import grade_essay, score_essays  # noqa: E402
from grade_retention import grade_item  # noqa: E402
from run_round import GpuSampler, gate, kv, profile, summarize  # noqa: E402

# One token cap for every short item, so each arm needs one engine load for
# them (greedy: the first 16 tokens are the same as at the quiz's cap of 16,
# and first-line grading is unchanged); the cap is what lets a model that
# does not stop show it.
CAPS = {"quiz": 48, "para": 48, "multi": 48, "pass": 48, "calib": 48, "long": 48, "instr": 48}
STOP = {2, 11, 12}


def parse_logit(line: str) -> dict:
    d = kv(line)
    top = []
    for t in str(d.get("top", "")).split(","):
        if ":" in t:
            a, b = t.split(":", 1)
            top.append([int(a), float(b)])
    d["top"] = top
    return d


def launch(args, max_new: int):
    cmd = [args.engine, args.checkpoint, args.tokenizer, "--chat", "--max-new", str(max_new)]
    if args.cuda_pack:
        cmd += ["--cuda-pack", args.cuda_pack]
    cmd += args.engine_args
    env = dict(os.environ)
    env["NEMO_LOGIT_TOPK"] = str(args.topk)
    for e in args.env:
        k, v = e.split("=", 1)
        env[k] = v
    flags = 0x00004000 if os.name == "nt" else 0  # BELOW_NORMAL_PRIORITY_CLASS
    err = open(args.stderr_log, "a", encoding="utf-8") if args.stderr_log else subprocess.DEVNULL
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=err, text=True,
                            encoding="utf-8", errors="replace", bufsize=1, creationflags=flags, env=env)
    ready, hw = "", ""
    while not ready:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited before READY")
        line = line.rstrip("\r\n")
        if line.startswith("READY"):
            ready = line
        elif line.startswith("ERROR"):
            raise RuntimeError(line)
    if args.cuda_pack:
        line = proc.stdout.readline().rstrip("\r\n")
        if line.startswith("HW "):
            hw = line
    return proc, cmd, ready, hw


def turn(proc, prompt: str) -> dict:
    proc.stdin.write("PROMPT " + prompt.replace("\n", "\\n") + "\n")
    proc.stdin.flush()
    pieces, logits, prefill, stats, mem, error, stat_lines = [], [], {}, {}, {}, "", []
    while True:
        line = proc.stdout.readline()
        if line == "":
            raise RuntimeError("engine exited mid-turn")
        line = line.rstrip("\r\n")
        if line.startswith("TOK "):
            pieces.append(line[4:])
        elif line.startswith("LOGIT "):
            logits.append(parse_logit(line))
        elif line.startswith("PREFILL "):
            prefill = kv(line)
        elif line.startswith("STATS "):
            stats = kv(line)
        elif line.startswith("MEM "):
            mem = kv(line)
        elif line.startswith("STAT "):
            stat_lines.append(line)
        elif line.startswith("ERROR"):
            error = line
            break
        elif line == "DONE":
            break
    return {"pieces": pieces, "text": answer_text(pieces), "logits": logits, "prefill": prefill,
            "stats": stats, "mem": mem, "error": error, "stat": stat_lines}


def termination(t: dict) -> dict:
    lg = t["logits"]
    natural = bool(lg) and lg[-1].get("tok") in STOP
    # the step that emitted the first newline after some answer text
    seen, at = False, None
    text = ""
    for k, p in enumerate(t["pieces"]):
        piece = answer_text(p)
        if "\n" in piece and seen:
            at = k
            break
        text += piece
        if text.strip():
            seen = True
    out = {"natural_stop": natural, "gen_steps": len(lg)}
    if at is not None and at < len(lg):
        s = lg[at]
        out.update(eol_step=at, eol_stop_lp=s.get("stop_lp"), eol_stop_rank=s.get("stop_rank"))
    elif natural:
        s = lg[-1]
        out.update(eol_step=len(lg) - 1, eol_stop_lp=s.get("stop_lp"), eol_stop_rank=s.get("stop_rank"))
    if lg:
        out["first_lp"] = lg[0].get("lp")
        out["first_margin"] = lg[0].get("margin")
        out["first_top"] = lg[0].get("top")
    return out


def run_group(args, items, max_new, sampler, grader, label):
    t_start = time.perf_counter()
    proc, cmd, ready, hw = None, None, "", ""
    rows, pauses, load_s, launches = [], [], 0.0, 0
    try:
        for it in items:
            rc, why = gate(args)
            if rc != 0:
                if proc is not None:
                    proc.stdin.write("QUIT\n"); proc.stdin.flush(); proc.wait(timeout=60)
                proc = None
                p0 = time.time()
                print("%s %s paused: %s" % (label, it["id"], why), flush=True)
                while rc != 0:
                    time.sleep(args.gate_poll)
                    rc, _ = gate(args)
                pauses.append({"before": it["id"], "reason": why, "s": round(time.time() - p0, 1)})
            if proc is None:
                t0 = time.perf_counter()
                proc, cmd, ready, hw = launch(args, max_new)
                load_s += time.perf_counter() - t0
                launches += 1
            elif rows:
                proc.stdin.write("RESET\n"); proc.stdin.flush()
            t0 = time.perf_counter()
            t = turn(proc, it["prompt"])
            t1 = time.perf_counter()
            row = {"id": it["id"], "set": it.get("set", "essay"), "prompt": it["prompt"], "text": t["text"], "error": t["error"]}
            row.update(grader(it, t["text"]))
            row["termination"] = termination(t)
            row["logits"] = t["logits"]
            row["profile"] = profile(t, t1 - t0)
            row["profile"]["lane"] = t["stats"].get("lane")
            row["profile"].update(sampler.window(t0, t1))
            rows.append(row)
            pr = row["profile"]
            print("%s %-10s %-5s stop=%s prefill=%s decode=%s %s" % (
                label, it["id"], row["grade"], int(row["termination"]["natural_stop"]), pr["prefill_tps"],
                pr["decode_tps"], (row.get("first_line") or row.get("topic") or "")[:50]), flush=True)
        if proc is not None:
            proc.stdin.write("QUIT\n"); proc.stdin.flush()
            try:
                proc.wait(timeout=60)
            except Exception:
                proc.kill()
            proc = None
    finally:
        if proc is not None and proc.poll() is None:
            proc.kill()
    return {"command": [Path(c).name if os.sep in c or "/" in c else c for c in cmd] if cmd else None,
            "ready": ready, "hw": hw, "max_new": max_new, "load_s": round(load_s, 3), "launches": launches,
            "pauses": pauses, "wall_s": round(time.perf_counter() - t_start, 3),
            "gpu": sampler.window(t_start, time.perf_counter()), "items": rows, "summary": summarize(rows)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--checkpoint", default=os.environ.get("NEMO_CKPT"))
    ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--cuda-pack", default=os.environ.get("NEMO_CUDA_PACK"))
    ap.add_argument("--battery", type=Path, default=HERE / "battery.json")
    ap.add_argument("--essays-file", type=Path, default=QUIZ / "essays.json")
    ap.add_argument("--sets", default="quiz,para,multi,pass,calib,instr,long")
    ap.add_argument("--essays", action="store_true")
    ap.add_argument("--essay-max-new", type=int, default=320)
    ap.add_argument("--ids", default="")
    ap.add_argument("--arm", required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--stderr-log", default=None)
    ap.add_argument("--gate-cmd", default="")
    ap.add_argument("--gate-poll", type=float, default=60.0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--engine-args", nargs=argparse.REMAINDER, default=[])
    args = ap.parse_args(argv)
    if not args.tokenizer:
        args.tokenizer = str(Path(args.checkpoint) / "tokenizer.json")
    if args.engine_args and args.engine_args[0] == "--":
        args.engine_args = args.engine_args[1:]
    bat = json.loads(args.battery.read_text(encoding="utf-8"))
    want = set(args.sets.split(",")) if args.sets else set()
    ids = set(args.ids.split(",")) if args.ids else None
    sel = [it for it in bat["items"] if it.get("gen", True) and it["set"] in want and (ids is None or it["id"] in ids)]
    groups = {}
    for it in sel:
        groups.setdefault(CAPS[it["set"]], []).append(it)
    sampler = GpuSampler().start()
    payload = {"arm": args.arm, "started": time.strftime("%Y-%m-%d %H:%M:%S %z"), "env": args.env,
               "topk": args.topk, "groups": {}}
    t0 = time.perf_counter()
    try:
        for cap in sorted(groups):
            ph = run_group(args, groups[cap], cap, sampler, grade_item, "G%d" % cap)
            payload["groups"][str(cap)] = ph
        if args.essays:
            ez = json.loads(args.essays_file.read_text(encoding="utf-8"))
            pf = ez.get("pass_fraction", 0.7)
            ph = run_group(args, ez["essays"], args.essay_max_new, sampler, lambda e, txt: grade_essay(e, txt, pf), "E")
            ph["score"] = score_essays(ph["items"])
            payload["essays"] = ph
            print("ESSAYS", ph["score"]["score"], flush=True)
    finally:
        sampler.stop()
    payload["wall_s"] = round(time.perf_counter() - t0, 3)
    payload["gpu_whole_run"] = sampler.window(0, float("inf"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("wrote", args.out, "wall=%.1fs" % payload["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
