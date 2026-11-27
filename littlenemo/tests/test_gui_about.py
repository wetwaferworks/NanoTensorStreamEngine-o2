#!/usr/bin/env python3
"""test_gui_about.py - the About/credits text and its agreement with the docs.

Checks the credits order (JD, then Cortana, Elfie, Miss Archie, then the
rest), the required Cortana / Grok Bot wording, and that README.md (credits
section last) and docs/CREDITS.md list the same people in the same order.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(HERE, "..", "gui"))

import nemo_about as about  # noqa: E402

LINE = "Co-implementation by Cortana, built with Grok Bot by SpaceX"
ORDER = ("JD", "Cortana", "Elfie", "Miss Archie", "ChatGPT (Gizmo)",
         "CloudAtlas", "Grok Bot")
fails = []


def expect(cond, name):
    print(("ok   " if cond else "FAIL ") + name)
    if not cond:
        fails.append(name)


def positions(text, starts):
    out = []
    for s in starts:
        i = text.find(s)
        out.append(i)
    return out


def ordered(pos):
    return all(p >= 0 for p in pos) and pos == sorted(pos)


def main():
    heads = [h for h, _ in about.CREDITS]
    expect(all(heads[i].startswith(ORDER[i]) for i in range(len(ORDER))),
           "About heads: JD, Cortana, Elfie, Miss Archie, then the rest")
    expect(LINE in about.credits_text(), "About carries the Cortana line")

    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as f:
        readme = f.read()
    h2 = [l for l in readme.splitlines() if l.startswith("## ")]
    expect(h2[-1] == "## Credits", "README: Credits is the last section")
    sec = readme[readme.index("\n## Credits"):]
    expect(LINE in sec, "README credits carry the Cortana line")
    marks = ["- **JD", "- **Cortana", "- **Elfie", "- **Miss Archie",
             "- **ChatGPT (Gizmo)", "- **CloudAtlas", "- **Grok Bot"]
    expect(ordered(positions(sec, marks)), "README credits order")

    with open(os.path.join(ROOT, "docs", "CREDITS.md"), encoding="utf-8") as f:
        cr = f.read()
    marks = ["### JD", "### Cortana", "### Elfie", "### Miss Archie",
             "### ChatGPT (Gizmo)", "### CloudAtlas", "### Grok Bot"]
    expect(ordered(positions(cr, marks)), "docs/CREDITS.md order")
    expect(LINE in cr, "docs/CREDITS.md carries the Cortana line")

    print("RESULT:", "FAILURES" if fails else "ALL PASS")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
