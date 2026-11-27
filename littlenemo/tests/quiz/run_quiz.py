#!/usr/bin/env python3
"""Run the Little Nemo 10-question smoke quiz against a live engine.

Checkpoint and tokenizer paths are arguments or NEMO_CKPT / NEMO_TOKENIZER.
This script never bakes in a machine-specific model path.

Typical raw-CPU reproduction (chat template on, thinking off, greedy,
16 new tokens):

    python littlenemo/tests/quiz/run_quiz.py \\
        --engine littlenemo/bin/engine_gui.exe \\
        --checkpoint /path/to/nemotron-nano-9b-v2 \\
        --tokenizer /path/to/nemotron-nano-9b-v2/tokenizer.json

The CUDA pack path uses nemo-engine-cuda.exe and also needs --cuda-pack
/path/to/the.nq2 (plus whatever expand flags you are measuring).

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from grade import answer_text, grade_question, score_grades  # noqa: E402


def load_questions(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_cmd(args: argparse.Namespace) -> list[str]:
    cmd = [args.engine, args.checkpoint, args.tokenizer]
    if not args.no_chat:
        cmd.append("--chat")
    # Thinking stays off unless the caller asked for --think.
    if args.think:
        cmd.append("--think")
    cmd.extend(["--max-new", str(args.max_new)])
    if args.threads is not None:
        cmd.extend(["--threads", str(args.threads)])
    if args.simd:
        cmd.append("--simd")
    if args.cuda_pack:
        cmd.extend(["--cuda-pack", args.cuda_pack])
    if args.engine_args:
        cmd.extend(args.engine_args)
    return cmd


def _readline(proc) -> str | None:
    line = proc.stdout.readline()
    if line == "":
        return None
    return line.rstrip("\r\n")


def wait_ready(proc) -> str:
    while True:
        line = _readline(proc)
        if line is None:
            raise RuntimeError("engine exited before READY")
        if line.startswith("READY"):
            return line
        if line.startswith("ERROR"):
            raise RuntimeError(line)


def run_turn(proc, prompt: str, raw_log) -> dict:
    proc.stdin.write("PROMPT " + prompt.replace("\n", "\\n") + "\n")
    proc.stdin.flush()
    pieces: list[str] = []
    think: list[str] = []
    stats = ""
    error = ""
    while True:
        line = _readline(proc)
        if line is None:
            raise RuntimeError("engine exited mid-turn")
        raw_log.append(line)
        if line.startswith("TOK "):
            pieces.append(line[4:])
        elif line.startswith("THINK "):
            think.append(line[6:])
        elif line.startswith("STATS"):
            stats = line
        elif line.startswith("ERROR"):
            error = line
            break
        elif line == "DONE":
            break
        # LAYER / TICK / PREFILL / DTC / EXPAND and friends are ignored.
    return {
        "pieces": pieces,
        "answer": answer_text(pieces),
        "think": answer_text(think),
        "stats": stats,
        "error": error,
    }


def parse_tps(stats: str) -> float | None:
    if not stats:
        return None
    for part in stats.split():
        if part.startswith("tps="):
            try:
                return float(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", help="path to engine_gui or nemo-engine-cuda")
    ap.add_argument("--checkpoint", default=os.environ.get("NEMO_CKPT"),
                    help="raw checkpoint directory, or NEMO_CKPT")
    ap.add_argument("--tokenizer", default=os.environ.get("NEMO_TOKENIZER"),
                    help="tokenizer.json, or NEMO_TOKENIZER (defaults to <checkpoint>/tokenizer.json)")
    ap.add_argument("--cuda-pack", default=os.environ.get("NEMO_CUDA_PACK"),
                    help="optional .nq2 pack for the CUDA engine")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.json")
    ap.add_argument("--max-new", type=int, default=16)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--simd", action="store_true")
    ap.add_argument("--think", action="store_true", help="leave thinking on (not the published record)")
    ap.add_argument("--no-chat", action="store_true", help="do not pass --chat (not the published record)")
    ap.add_argument("--engine-args", nargs=argparse.REMAINDER, default=[],
                    help="extra engine flags after --")
    ap.add_argument("--out", type=Path, default=None, help="write a JSON result file here")
    args = ap.parse_args(argv)

    if not args.engine or not args.checkpoint:
        ap.print_help()
        print("\nNeed --engine and --checkpoint (or NEMO_CKPT). "
              "No model path is built in.", file=sys.stderr)
        return 2
    if not args.tokenizer:
        guess = Path(args.checkpoint) / "tokenizer.json"
        args.tokenizer = str(guess)

    quiz = load_questions(args.questions)
    qs = quiz["questions"]
    # Drop a leading -- from remainder-style flags.
    if args.engine_args and args.engine_args[0] == "--":
        args.engine_args = args.engine_args[1:]

    import subprocess
    cmd = build_cmd(args)
    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    raw_log = []
    try:
        ready = wait_ready(proc)
        raw_log.append(ready)
        print(ready)
        rows = []
        first = True
        for q in qs:
            if not first:
                proc.stdin.write("RESET\n")
                proc.stdin.flush()
                raw_log.append("RESET")
            first = False
            raw_log.append("PROMPT " + q["prompt"])
            turn = run_turn(proc, q["prompt"], raw_log)
            g = grade_question(q, turn["answer"])
            tps = parse_tps(turn["stats"])
            row = {
                "id": q["id"],
                "prompt": q["prompt"],
                "expect": q["expect"],
                "first_line": g["first_line"],
                "grade": g["grade"],
                "answer": turn["answer"],
                "stats": turn["stats"],
                "tps": tps,
                "error": turn["error"],
            }
            rows.append(row)
            tps_s = "  tps=%.3f" % tps if tps is not None else ""
            print("Q%-2d  %-5s  %s%s" % (q["id"], g["grade"], g["first_line"] or "(empty)", tps_s))
        proc.stdin.write("QUIT\n")
        proc.stdin.flush()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
    finally:
        if proc.poll() is None:
            proc.kill()

    scored = score_grades(rows)
    elapsed = time.perf_counter() - t0
    tps_vals = [r["tps"] for r in rows if r["tps"] is not None]
    mean_tps = sum(tps_vals) / len(tps_vals) if tps_vals else None
    print("SCORE %s  (%d pass, %d hedge, %d fail)  wall=%.1fs%s" % (
        scored["score"], scored["passes"], scored["hedges"], scored["fails"],
        elapsed,
        ("  mean_tps=%.3f" % mean_tps) if mean_tps is not None else "",
    ))
    payload = {
        "quiz": quiz.get("name"),
        "command": cmd,
        "ready": raw_log[0] if raw_log else "",
        "score": scored,
        "mean_tps": mean_tps,
        "wall_s": elapsed,
        "questions": rows,
    }
    # Strip argv paths from being required: they are in command for the
    # operator's own log, which they choose not to commit.
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("wrote", args.out)
    return 0 if scored["fails"] == 0 and scored["hedges"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
