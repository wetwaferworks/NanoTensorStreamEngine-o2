#!/usr/bin/env python3
"""Unit tests for the 50-question set and the essay rubric. No checkpoint required.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from grade import grade_question, score_grades  # noqa: E402
from grade_essay import grade_essay, score_essays  # noqa: E402


def main() -> int:
    failures, checks = [], []

    def expect(name, cond):
        checks.append(name)
        print(("ok  " if cond else "FAIL"), name)
        if not cond:
            failures.append(name)

    qz = json.loads((HERE / "questions50.json").read_text(encoding="utf-8"))
    qs = qz["questions"]
    expect("fifty questions", len(qs) == 50)
    expect("ids 1..50", [q["id"] for q in qs] == list(range(1, 51)))
    expect("conditions match the record", qz["conditions"]["max_new"] == 16
           and qz["conditions"]["thinking"] is False and qz["conditions"]["chat_template"] is True)
    expect("prompts are unique", len({q["prompt"] for q in qs}) == 50)
    byid = {q["id"]: q for q in qs}

    # Every gold answer and every alias passes; every listed wrong answer fails.
    golds = [grade_question(q, q["expect"]) for q in qs]
    expect("bare gold answers score 50/50", score_grades(golds)["score"] == "50/50")
    for q in qs:
        for a in q.get("accept", []):
            expect("Q%d alias %r passes" % (q["id"], a), grade_question(q, "It is %s." % a)["grade"] == "pass")
        for w in q.get("wrong", []):
            expect("Q%d wrong %r fails" % (q["id"], w), grade_question(q, "%s." % w)["grade"] == "fail")
        # No gold answer is already in its own prompt, so an echo cannot pass.
        cs = q.get("case_sensitive", False)
        pat = r"(?<![A-Za-z0-9])" + re.escape(q["expect"]) + r"(?![A-Za-z0-9])"
        expect("Q%d gold not in prompt" % q["id"], re.search(pat, q["prompt"], 0 if cs else re.I) is None)

    # Sentence answers and wrong-first lines.
    he = next(q for q in qs if q["expect"] == "He")
    expect("helium: sentence with He passes", grade_question(he, "The chemical symbol for helium is He.")["grade"] == "pass")
    expect("helium: pronoun he does not pass", grade_question(he, "he said H")["grade"] == "fail")
    jup = next(q for q in qs if q["expect"] == "Jupiter")
    expect("Earth before Jupiter fails", grade_question(jup, "Earth, then Jupiter")["grade"] == "fail")
    can = next(q for q in qs if q["expect"] == "Ottawa")
    expect("Toronto fails", grade_question(can, "Toronto")["grade"] == "fail")
    expect("Q1 sentence passes", grade_question(byid[1], "12 times 12 is 144.")["grade"] == "pass")
    expect("Q1 wrong number fails", grade_question(byid[1], "12 times 12 is 154.")["grade"] == "fail")

    # Essays.
    ez = json.loads((HERE / "essays.json").read_text(encoding="utf-8"))
    es = ez["essays"]
    expect("ten essays", len(es) == 10)
    expect("every essay has required facts", all(len(e["required"]) >= 5 for e in es))
    expect("every essay has forbidden claims", all(len(e.get("forbidden", [])) >= 1 for e in es))
    ok = True
    for e in es:
        for r in e["required"] + e.get("forbidden", []):
            for p in r["any"]:
                try:
                    re.compile(p, re.I)
                except re.error:
                    ok = False
    expect("all rubric patterns compile", ok)

    photo = es[0]
    good = ("Photosynthesis is how plants make food. In the chloroplasts, chlorophyll absorbs sunlight. "
            "The plant takes in carbon dioxide from the air and water from the soil and turns them into "
            "glucose, a sugar, releasing oxygen.")
    g = grade_essay(photo, good, ez["pass_fraction"])
    expect("good photosynthesis essay hits 7/7", g["required_hit"] == 7 and g["grade"] == "pass")
    bad = good + " Photosynthesis takes place in the mitochondria."
    g = grade_essay(photo, bad, ez["pass_fraction"])
    expect("forbidden claim fails the essay", g["grade"] == "fail" and len(g["forbidden_hits"]) == 1)
    g = grade_essay(photo, "Plants are green.", ez["pass_fraction"])
    expect("empty-of-facts essay fails", g["grade"] == "fail" and g["required_hit"] == 0)
    seasons = next(e for e in es if e["topic"] == "seasons")
    g = grade_essay(seasons, "Seasons are caused by the distance from the Sun.", 0.7)
    expect("distance myth is flagged", len(g["forbidden_hits"]) == 1)
    s = score_essays([grade_essay(photo, good, 0.7), grade_essay(photo, bad, 0.7)])
    expect("essay score counts passes", s["score"] == "1/2" and s["forbidden_total"] == 1)

    print("RESULT: %d checks, %d failures" % (len(checks), len(failures)))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
