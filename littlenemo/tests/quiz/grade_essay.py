"""Grade a Little Nemo essay against a fixed fact rubric (essays.json).

A required fact is present when any of its case-insensitive regular
expressions matches the essay text. A forbidden claim is flagged when
any of its patterns matches. Score is required facts present over
required facts; an essay passes at the rubric's pass_fraction (0.7)
with no forbidden claim. Nothing here judges style, only checkable facts.

Direction by JD. Co-implementation by Cortana, built with Grok Bot
by SpaceX.
"""
from __future__ import annotations

import re
from typing import Any


def _hits(text: str, patterns) -> list[str]:
    out = []
    for p in patterns:
        m = re.search(p, text, re.I)
        if m:
            out.append(m.group(0))
    return out


def grade_essay(essay: dict[str, Any], text: str, pass_fraction: float = 0.7) -> dict[str, Any]:
    present, missing, flagged = [], [], []
    for r in essay["required"]:
        h = _hits(text, r["any"])
        (present if h else missing).append({"fact": r["fact"], "matched": h[:1]})
    for f in essay.get("forbidden", []):
        h = _hits(text, f["any"])
        if h:
            flagged.append({"claim": f["claim"], "matched": h[:1]})
    n = len(essay["required"])
    frac = len(present) / n if n else 0.0
    return {
        "id": essay["id"],
        "topic": essay.get("topic", ""),
        "required_n": n,
        "required_hit": len(present),
        "fraction": round(frac, 4),
        "present": present,
        "missing": missing,
        "forbidden_hits": flagged,
        "grade": "pass" if frac >= pass_fraction and not flagged else "fail",
    }


def score_essays(rows) -> dict[str, Any]:
    hit = sum(r["required_hit"] for r in rows)
    tot = sum(r["required_n"] for r in rows)
    return {
        "n": len(rows),
        "passes": sum(1 for r in rows if r["grade"] == "pass"),
        "facts_hit": hit,
        "facts_total": tot,
        "forbidden_total": sum(len(r["forbidden_hits"]) for r in rows),
        "score": f"{sum(1 for r in rows if r['grade'] == 'pass')}/{len(rows)}",
    }
