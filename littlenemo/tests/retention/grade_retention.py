"""Graders for the NanoTensor retention battery (battery.json).

Short answers use the quiz grader (littlenemo/tests/quiz/grade.py, first
answer line). Unanswerable items pass when the first line abstains;
answerable controls under the same instruction pass only when answered
and not abstained. Instruction items are checked by rule twice: strict
(the whole reply must satisfy the instruction, so text after the answer
counts against it) and first-line (only the first answer line), which
separates "did it follow the instruction" from "did it stop".

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

QUIZ = Path(__file__).resolve().parent.parent / "quiz"
if str(QUIZ) not in sys.path:
    sys.path.insert(0, str(QUIZ))
from grade import answer_text, first_answer_line, grade_question  # noqa: E402

ABSTAIN = re.compile(r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know|not\s+sure|unknown|no\s+(such|real|known|record)|"
                     r"does\s+not\s+exist|doesn'?t\s+exist|not\s+a\s+real|fictional|made[- ]up|cannot\s+(answer|find|determine)|"
                     r"there\s+is\s+no|not\s+aware)\b", re.I)


def _check(chk: dict, text: str) -> bool:
    t = text.strip()
    if chk.get("lower") and t != t.lower():
        return False
    if "exact" in chk:
        a = t.rstrip(".") if not chk["exact"].endswith(".") else t
        if a != chk["exact"]:
            return False
    if "lines" in chk:
        lines = [x for x in t.splitlines() if x.strip()]
        if len(lines) != chk["lines"]:
            return False
    if "words" in chk:
        if len(re.findall(r"[A-Za-z']+", t)) != chk["words"]:
            return False
    if "any" in chk and not any(re.search(r"\b%s\b" % re.escape(w), t, re.I) for w in chk["any"]):
        return False
    if "json" in chk:
        try:
            if json.loads(t) != chk["json"]:
                return False
        except Exception:
            return False
    if "regex" in chk and not re.match(chk["regex"], t):
        return False
    if "starts" in chk and not t.startswith(chk["starts"]):
        return False
    if "ends" in chk and not t.endswith(chk["ends"]):
        return False
    return True


def first_block(text: str) -> str:
    """The reply up to its first blank line (the answer block)."""
    t = answer_text(text).strip()
    return re.split(r"\n\s*\n", t, maxsplit=1)[0] if t else ""


def grade_item(it: dict, text: str) -> dict:
    text = answer_text(text)
    line = first_answer_line(text)
    out = {"first_line": line}
    s = it["set"]
    if s == "instr":
        strict = _check(it["check"], text)
        # first-line view: for multi-line instructions, the first answer block
        loose_src = first_block(text) if it["check"].get("lines", 1) > 1 else line
        loose = _check(it["check"], loose_src)
        out.update(grade="pass" if strict else "fail", strict=strict, first_line_ok=loose)
        return out
    if s == "calib" and it.get("abstain"):
        ab = bool(ABSTAIN.search(line))
        out.update(grade="pass" if ab else "fail", abstained=ab)
        return out
    g = grade_question(dict(it, kind="short-answer"), text)
    out["grade"] = g["grade"]
    if s == "calib":
        ab = bool(ABSTAIN.search(line))
        out["abstained"] = ab
        if ab:
            out["grade"] = "fail"
    out["clean_stop"] = None  # filled by the runner from the engine's stop
    return out
