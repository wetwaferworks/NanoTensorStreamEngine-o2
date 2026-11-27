#!/usr/bin/env python3
"""Unit tests for the Little Nemo quiz grader. No checkpoint required.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from grade import (  # noqa: E402
    answer_text,
    first_answer_line,
    grade_question,
    score_grades,
)


def expect(name, cond, failures, checks):
    checks.append(name)
    if cond:
        print("ok  ", name)
    else:
        failures.append(name)
        print("FAIL", name)


def main() -> int:
    failures = []
    checks = []
    quiz = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))
    qs = {q["id"]: q for q in quiz["questions"]}

    expect("ten questions", len(qs) == 10, failures, checks)
    expect("max_new is 16", quiz["conditions"]["max_new"] == 16, failures, checks)
    expect("thinking off", quiz["conditions"]["thinking"] is False, failures, checks)
    expect("chat template on", quiz["conditions"]["chat_template"] is True, failures, checks)

    # Protocol unescape: TOK pieces with escaped newlines.
    text = answer_text(["True", "\\n", "or leftover"])
    expect("unescape joins pieces", text == "True\nor leftover", failures, checks)
    expect("first line skips nothing needed", first_answer_line("\\n  True.\\n") == "True.", failures, checks)

    # Published packed first lines from ACCEL-CUDA.md.
    pack_grades = []
    for q in quiz["questions"]:
        g = grade_question(q, q["published_pack_first_line"])
        pack_grades.append(g)
        expect(
            "pack Q%d grades as %s" % (q["id"], q["published_pack_grade"]),
            g["grade"] == q["published_pack_grade"],
            failures,
            checks,
        )
    scored = score_grades(pack_grades)
    expect("pack headline is 7/10", scored["score"] == "7/10", failures, checks)
    expect("pack 7 pass, 2 hedge, 1 fail",
           scored["passes"] == 7 and scored["hedges"] == 2 and scored["fails"] == 1,
           failures, checks)

    # Raw 10/10: clean expected answers, including a trailing period.
    raw_grades = []
    for q in quiz["questions"]:
        clean = q["expect"] + ("." if q["kind"] == "true-false" else "")
        g = grade_question(q, clean)
        raw_grades.append(g)
        expect("raw-clean Q%d passes" % q["id"], g["grade"] == "pass", failures, checks)
    expect("raw clean is 10/10", score_grades(raw_grades)["score"] == "10/10", failures, checks)

    # Wrong first word is a fail, even if Jupiter appears later.
    g = grade_question(qs[10], "Earth is larger than Jupiter")
    expect("Earth-first is a fail", g["grade"] == "fail", failures, checks)
    g = grade_question(qs[10], "Jupiter, not Earth")
    expect("Jupiter-first is a pass", g["grade"] == "pass", failures, checks)

    # Empty answer fails.
    g = grade_question(qs[1], "   \n")
    expect("empty answer fails", g["grade"] == "fail", failures, checks)

    # True/False swapped.
    g = grade_question(qs[1], "False")
    expect("wrong T/F fails", g["grade"] == "fail", failures, checks)
    g = grade_question(qs[2], "True or False")
    expect("wrong-first hedge is a fail", g["grade"] == "fail", failures, checks)

    print("RESULT: %d checks, %d failures" % (len(checks), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
