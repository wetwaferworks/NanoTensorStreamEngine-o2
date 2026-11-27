"""Grade a Little Nemo quiz answer from the first visible line.

The 2026-11-26 record (docs/ACCEL-CUDA.md) grades the first answer line
of a greedy, chat-templated, no-think turn capped at 16 new tokens.
Headline score counts only clean passes; a hedge is the right word
first with extra hedging on that line (for example "True or False");
a fail is a wrong or missing answer.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import re
from typing import Any


def unescape_piece(piece: str) -> str:
    """Turn protocol-escaped text into ordinary characters."""
    out = []
    i = 0
    while i < len(piece):
        if piece[i] == "\\" and i + 1 < len(piece):
            nxt = piece[i + 1]
            if nxt == "n":
                out.append("\n")
                i += 2
                continue
            if nxt == "t":
                out.append("\t")
                i += 2
                continue
            if nxt == "\\":
                out.append("\\")
                i += 2
                continue
        out.append(piece[i])
        i += 1
    return "".join(out)


def answer_text(pieces) -> str:
    if isinstance(pieces, str):
        return unescape_piece(pieces)
    return "".join(unescape_piece(p) for p in pieces)


def first_answer_line(text: str) -> str:
    for line in answer_text(text).splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def _word_re(word: str, case_sensitive: bool = False) -> re.Pattern:
    flags = 0 if case_sensitive else re.I
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(word) + r"(?![A-Za-z0-9])", flags)


def _first_match_pos(line: str, words, case_sensitive: bool = False) -> tuple[int, str] | tuple[None, None]:
    best_pos = None
    best_word = None
    for w in words:
        m = _word_re(w, case_sensitive).search(line)
        if m and (best_pos is None or m.start() < best_pos):
            best_pos = m.start()
            best_word = w
    return best_pos, best_word


def _clean_true_false(line: str, expect: str) -> bool:
    return re.fullmatch(re.escape(expect) + r"\.?", line, re.I) is not None


def grade_question(q: dict[str, Any], text: str) -> dict[str, Any]:
    line = first_answer_line(text)
    expect = str(q["expect"])
    kind = q.get("kind", "short-answer")
    wrong = [str(w) for w in q.get("wrong", [])]
    result = {"id": q["id"], "first_line": line, "grade": "fail"}

    if not line:
        return result

    if kind == "true-false":
        opposite = "False" if expect.lower() == "true" else "True"
        pos_ok, _ = _first_match_pos(line, [expect])
        pos_bad, _ = _first_match_pos(line, [opposite])
        if pos_ok is None:
            return result
        if pos_bad is not None and pos_bad < pos_ok:
            return result
        if _clean_true_false(line, expect) and pos_bad is None:
            result["grade"] = "pass"
        else:
            result["grade"] = "hedge"
        return result

    # accept lists other spellings of the same gold answer ("1,024",
    # "seven"); case_sensitive keeps element symbols like "He" from
    # matching ordinary words.
    cs = bool(q.get("case_sensitive", False))
    golds = [expect] + [str(a) for a in q.get("accept", [])]
    pos_ok, _ = _first_match_pos(line, golds, cs)
    pos_bad, bad = _first_match_pos(line, wrong, cs)
    if pos_ok is None:
        return result
    if pos_bad is not None and (pos_ok is None or pos_bad < pos_ok):
        return result
    # A short-answer pass allows a short sentence that leads with the
    # expected word (optional period). Extra hedging after a correct
    # capital or number is still a pass, matching the published table.
    result["grade"] = "pass"
    return result


def score_grades(grades) -> dict[str, Any]:
    n = len(grades)
    passes = sum(1 for g in grades if g["grade"] == "pass")
    hedges = sum(1 for g in grades if g["grade"] == "hedge")
    fails = sum(1 for g in grades if g["grade"] == "fail")
    return {
        "n": n,
        "passes": passes,
        "hedges": hedges,
        "fails": fails,
        "score": f"{passes}/{n}",
    }
